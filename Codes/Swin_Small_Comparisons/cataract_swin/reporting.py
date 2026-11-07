"""Post-training exports; this module never changes or retrains the model."""
from pathlib import Path
from collections import Counter
import html
import io
import json
import shutil
import textwrap
from urllib.parse import quote
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
from .io_utils import save_json,file_hash
from .tracking import write_csv
from .preprocessing import FitPad, preprocessing_metadata

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def slug(value):
    return ''.join(c if c.isalnum() else '_' for c in value).strip('_')


def save_figure(fig,path):
    """Write complete PNG bytes atomically, including for large image galleries."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=155,bbox_inches='tight')
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(buffer.getvalue());temporary.replace(path)
    plt.close(fig)


def history_plots(folder):
    path=folder/'history.csv'
    if not path.exists():return
    history=pd.read_csv(path)
    if history.empty:return
    extra=pd.read_csv(folder/'epoch_metrics.csv') if (folder/'epoch_metrics.csv').exists() else None
    fig,axs=plt.subplots(2,2,figsize=(11,7.5))
    for prefix,label in [('train','Training'),('val','Validation')]:
        axs[0,0].plot(history.epoch,history[prefix+'_loss'],label=label)
        axs[0,1].plot(history.epoch,history[prefix+'_accuracy'],label=label)
        if extra is not None and prefix+'_macro_f1' in extra:
            axs[1,0].plot(extra.epoch,extra[prefix+'_macro_f1'],label=label)
    axs[0,0].set(title='Loss',ylabel='Loss');axs[0,1].set(title='Accuracy',ylabel='Accuracy',ylim=(0,1))
    axs[1,0].set(title='Macro-F1 of recorded epoch predictions',ylabel='Macro-F1',ylim=(0,1))
    if extra is None:axs[1,0].text(.5,.5,'No extra epoch records',ha='center',va='center',transform=axs[1,0].transAxes)
    for key in ['head_lr','backbone_lr']:
        if key in history:axs[1,1].plot(history.epoch,history[key],label=key.replace('_',' '))
    axs[1,1].set(title='Learning rates used during each epoch',ylabel='Learning rate')
    for ax in axs.flat:
        ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=9)
    fig.suptitle('Swin-Small training history\nTraining metrics use the weighted, augmented training stream.',fontsize=13)
    fig.tight_layout();save_figure(fig,folder/'learning_curves.png')
    lr_cols=['epoch']+[c for c in ['head_lr','backbone_lr'] if c in history]
    write_csv(history[lr_cols],folder/'learning_rates.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    for c in lr_cols[1:]:ax.plot(history.epoch,history[c],label=c.replace('_',' '))
    ax.set(xlabel='Epoch',ylabel='Learning rate',title='Learning rates before the epoch scheduler update');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();save_figure(fig,folder/'learning_rate_curve.png')


def load_predictions(path,manifest,classes):
    predictions=pd.read_csv(path)
    if predictions.duplicated(['video_id','file_name']).any():raise ValueError('Repeated prediction identity.')
    split='val' if path.name.startswith('val_') else 'test'
    expected=manifest[manifest.split==split]
    keys=['file_name','video_id','label']
    result=predictions.merge(expected,on=keys,how='outer',validate='one_to_one',indicator=True)
    if not result['_merge'].eq('both').all():raise ValueError(f'{split} predictions do not cover their manifest exactly.')
    result=result.drop(columns='_merge')
    p=result[[f'prob_{i}' for i in range(len(classes))]].to_numpy()
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5) or (p<0).any() or (p>1+1e-6).any():
        raise ValueError('Invalid class probabilities.')
    if not np.array_equal(p.argmax(1),result.prediction):raise ValueError('Predicted labels disagree with saved probabilities.')
    result['true_class']=result.label.map(dict(enumerate(classes)))
    result['predicted_class']=result.prediction.map(dict(enumerate(classes)))
    result['confidence']=p.max(1)
    result['true_class_probability']=p[np.arange(len(result)),result.label.to_numpy(dtype=int)]
    result['correct']=result.label==result.prediction
    result['entropy_nats']=-(p*np.log(np.clip(p,1e-12,1))).sum(1)
    top=np.sort(p,axis=1)
    result['top_two_probability_margin']=top[:,-1]-top[:,-2]
    return result


def confusion_outputs(frame,classes,folder):
    matrix=confusion_matrix(frame.label,frame.prediction,labels=range(len(classes)))
    norm=np.divide(matrix,matrix.sum(1,keepdims=True),out=np.full(matrix.shape,np.nan),where=matrix.sum(1,keepdims=True)>0)
    pd.DataFrame(matrix,index=classes,columns=classes).rename_axis('reference_class').to_csv(folder/'confusion_counts.csv')
    pd.DataFrame(norm,index=classes,columns=classes).rename_axis('reference_class').to_csv(folder/'confusion_row_normalized.csv')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    ticks=[textwrap.fill(c,15) for c in classes]
    for ax,values,title in zip(axs,[matrix,norm],['Counts','Row-normalized proportions']):
        ax.imshow(values,cmap='Blues')
        for i in range(len(classes)):
            for j in range(len(classes)):
                value=values[i,j]
                label=str(matrix[i,j]) if title=='Counts' else (f'{value:.1%}' if np.isfinite(value) else 'n.a.')
                ax.text(j,i,label,ha='center',va='center',color='white' if value>np.nanmax(values)*.55 else 'black')
        ax.set_xticks(range(len(classes)),ticks,fontsize=9);ax.set_yticks(range(len(classes)),ticks,fontsize=9)
        ax.set(title=title,xlabel='Predicted class',ylabel='Reference class')
    fig.tight_layout();save_figure(fig,folder/'confusion_matrix.png')
    report=classification_report(frame.label,frame.prediction,labels=range(len(classes)),target_names=classes,output_dict=True,zero_division=0)
    save_json(folder/'classification_report.json',report)
    # Keep accuracy in the JSON/summary, not broadcast into precision/recall/support columns.
    pd.DataFrame({k:v for k,v in report.items() if isinstance(v,dict)}).T.rename_axis('class_or_average').to_csv(folder/'classification_report.csv')
    errors=frame[~frame.correct]
    pairs=errors.groupby(['label','prediction','true_class','predicted_class']).agg(
        frames=('file_name','size'),videos=('video_id','nunique'),mean_confidence=('confidence','mean')).reset_index()
    pairs=pairs.sort_values('frames',ascending=False)
    write_csv(pairs,folder/'confusion_pairs.csv')
    return report,pairs


def roc_outputs(frame,classes,folder):
    points=[];scores={};fig,ax=plt.subplots(figsize=(7,5))
    indices=[1] if len(classes)==2 else range(len(classes))
    for i in indices:
        target=(frame.label.to_numpy()==i).astype(int)
        if len(np.unique(target))<2:
            scores[classes[i]]=None;continue
        fpr,tpr,thresholds=roc_curve(target,frame[f'prob_{i}'].to_numpy())
        score=float(auc(fpr,tpr));scores[classes[i]]=score
        points.extend({'class_index':int(i),'class_name':classes[i],'fpr':float(x),'tpr':float(y),
                       'threshold':float(t) if np.isfinite(t) else None} for x,y,t in zip(fpr,tpr,thresholds))
        ax.plot(fpr,tpr,label=f'{classes[i]} (AUC {score:.4f})')
    ax.plot([0,1],[0,1],linestyle='--',color='gray')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC from the saved class probabilities',xlim=(0,1),ylim=(0,1.02))
    if points:ax.legend(loc='lower right',fontsize=9)
    ax.grid(alpha=.2);fig.tight_layout();save_figure(fig,folder/'roc.png')
    write_csv(pd.DataFrame(points,columns=['class_index','class_name','fpr','tpr','threshold']),folder/'roc_points.csv')
    result={'per_class_ovr_auc':scores,'macro_ovr_auc':float(np.mean(list(scores.values()))) if scores and all(v is not None for v in scores.values()) else None,
            'definition':'Binary: positive poor-dilation class. Multiclass: one-vs-rest, unweighted mean of per-class AUCs.'}
    save_json(folder/'roc_auc.json',result)
    return result


def video_summary(frame,classes,folder):
    result=[]
    for vid,g in frame.groupby('video_id',sort=True):
        wrong=g[~g.correct]
        row={'video_id':vid,'frames':len(g),'errors':len(wrong),'accuracy':float(g.correct.mean()),
             'labels_present':'; '.join(classes[i] for i in sorted(g.label.unique())),
             'mixed_reference_labels':g.label.nunique()>1,
             'highest_wrong_confidence':float(wrong.confidence.max()) if len(wrong) else None}
        for i in range(len(classes)):
            part=g[g.label==i];row[f'class_{i}_frames']=len(part);row[f'class_{i}_errors']=int((~part.correct).sum())
        result.append(row)
    write_csv(pd.DataFrame(result).sort_values(['errors','frames'],ascending=False),folder/'video_error_summary.csv')


def restore_review_fields(frame,path):
    columns=['reviewed_reason','reviewer_notes','label_review_status']
    old=pd.read_csv(path,keep_default_na=False) if path.exists() else None
    if old is not None and len(old):
        keys=['file_name','video_id']
        kept=old[keys+[c for c in columns if c in old]].drop_duplicates(keys)
        frame=frame.merge(kept,on=keys,how='left',validate='one_to_one')
    for c in columns:
        if c not in frame:frame[c]=''
        else:frame[c]=frame[c].fillna('')
    return frame


def copy_test_frames(frame,classes,folder,options,preprocessing=None):
    """Copy original bytes and exactly the same padded RGB view used by test."""
    frame=frame.copy()
    transform=FitPad(preprocessing)
    spec=preprocessing_metadata(preprocessing)
    save_json(folder/'test_images'/'preprocessing.json',spec)
    paths=[];inputs=[]
    for row in frame.itertuples():
        source=Path(row.image_path)
        parts=Path(f'true_{row.label}_{slug(classes[row.label])}')/f'pred_{row.prediction}_{slug(classes[row.prediction])}'/row.video_id
        # Preserve original filenames and video directories, rejecting path traversal.
        if Path(row.file_name).name!=row.file_name or Path(row.video_id).name!=row.video_id or row.video_id in {'.','..'}:
            raise ValueError('Unsafe image identity in export.')
        copied=folder/'test_images'/'originals'/parts/row.file_name
        if not source.is_file() and copied.is_file():source=copied
        if options['test_originals']:
            copied.parent.mkdir(parents=True,exist_ok=True)
            if not copied.exists() or file_hash(copied)!=row.image_sha256:
                if source.resolve()==copied.resolve():raise ValueError(f'Copied original is corrupted and source is missing: {row.file_name}')
                shutil.copy2(source,copied)
            if file_hash(copied)!=row.image_sha256:raise ValueError(f'Export image checksum mismatch: {row.file_name}')
            paths.append(copied.relative_to(folder).as_posix())
        else:paths.append('')
        if options['test_inputs']:
            # Prefer a verified retained original; never export a changed source
            # file as though it were the exact frame evaluated by the model.
            if options['test_originals']:
                source=copied  # Already verified immediately above.
            elif file_hash(source)!=row.image_sha256:
                raise ValueError(f'Model-input export source checksum mismatch: {row.file_name}')
            image_path=folder/'test_images'/'model_input_rgb'/parts/(row.file_name+'.png')
            image_path.parent.mkdir(parents=True,exist_ok=True)
            # Regenerate atomically instead of retaining stale stretched/corrupt views.
            # FitPad is the actual loader callable, not an independent approximation.
            with Image.open(source) as original:
                view=transform(original,source=getattr(row,'source',None))
                buffer=io.BytesIO();view.save(buffer,format='PNG')
            temporary=image_path.with_suffix('.png.tmp');temporary.write_bytes(buffer.getvalue());temporary.replace(image_path)
            inputs.append(image_path.relative_to(folder).as_posix())
        else:inputs.append('')
    frame['copied_original']=paths;frame['model_input_rgb']=inputs
    review_path=folder/'test_images'/'index.csv'
    frame=restore_review_fields(frame,review_path)
    write_csv(frame,review_path)
    make_gallery(frame,classes,folder/'test_gallery.html')
    return frame


def make_gallery(frame,classes,path):
    fields=['file_name','video_id','true_class','predicted_class','confidence','correct','copied_original','model_input_rgb']
    rows=json.loads(frame[fields].to_json(orient='records'))
    for r in rows:
        for key in ['copied_original','model_input_rgb']:
            if r[key]:r[key]=quote(r[key],safe='/')
    payload=json.dumps({'classes':classes,'rows':rows},ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e')
    template='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swin-Small test predictions</title><style>
body{font:15px system-ui,sans-serif;margin:24px;color:#172635;background:#f5f7fa}h1{font-size:24px}label{display:inline-block;margin:6px 16px 6px 0}select,input{padding:7px;border:1px solid #a9b7c6;border-radius:4px}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:16px}.card{background:white;padding:12px;border:1px solid #cbd4df;border-radius:7px}.wrong{border-top:4px solid #b34238}.right{border-top:4px solid #3f6c65}.card img{width:100%;height:245px;object-fit:contain;background:#121820}.meta{font-size:13px;overflow-wrap:anywhere;line-height:1.5}.image-label{font-size:12px;color:#596779}#count{margin:16px 0}.controls{position:sticky;top:0;background:#f5f7fa;padding:8px 0;z-index:2}a{color:#1e5a85}</style>
<h1>Test predictions from the validation-selected checkpoint</h1><p>Original frames and aspect-preserving padded model-input RGB views (before normalization). Reference labels are from the supplied metadata. Review causes in test_images/index.csv.</p>
<div class="controls"><label>Reference <select id="truth"></select></label><label>Prediction <select id="pred"></select></label><label>Outcome <select id="outcome"><option value="all">All</option><option value="wrong" selected>Errors</option><option value="right">Correct</option></select></label><label>Search <input id="search" placeholder="Video or filename"></label><label>Sort <select id="sort"><option value="confidence">Confidence: high first</option><option value="video">Video</option></select></label></div><div id="count"></div><div id="grid"></div>
<script id="data" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
for(const id of ['truth','pred']){const sel=document.getElementById(id);for(const name of ['All',...data.classes]){const option=document.createElement('option');option.value=name;option.textContent=name;sel.append(option)}}
function draw(){let rows=data.rows.filter(r=>(truth.value==='All'||r.true_class===truth.value)&&(pred.value==='All'||r.predicted_class===pred.value)&&(outcome.value==='all'||r.correct===(outcome.value==='right'))&&(r.video_id+' '+r.file_name).toLowerCase().includes(search.value.toLowerCase()));rows.sort(sort.value==='video'?(a,b)=>a.video_id.localeCompare(b.video_id):(a,b)=>b.confidence-a.confidence);document.getElementById('count').textContent=rows.length+' / '+data.rows.length+' test frames';const grid=document.getElementById('grid');grid.replaceChildren();for(const r of rows){const card=document.createElement('article');card.className='card '+(r.correct?'right':'wrong');const title=document.createElement('strong');title.textContent=r.true_class+' → '+r.predicted_class;card.append(title);for(const [key,label] of [['copied_original','Original image'],['model_input_rgb','Padded model-input RGB']]){if(!r[key])continue;const caption=document.createElement('div');caption.className='image-label';caption.textContent=label;card.append(caption);const link=document.createElement('a');link.href=r[key];link.target='_blank';const image=document.createElement('img');image.src=r[key];image.alt=label;image.loading='lazy';link.append(image);card.append(link)}const meta=document.createElement('p');meta.className='meta';meta.textContent='Confidence: '+(r.confidence*100).toFixed(2)+'% | '+r.video_id+' | '+r.file_name;card.append(meta);grid.append(card)}}
const truth=document.getElementById('truth'),pred=document.getElementById('pred'),outcome=document.getElementById('outcome'),search=document.getElementById('search'),sort=document.getElementById('sort');for(const id of ['truth','pred','outcome','search','sort'])document.getElementById(id).addEventListener('input',draw);draw();</script></html>'''
    path.write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')


def export_analysis(folder,manifest,classes,options,preprocessing=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    spec=preprocessing_metadata(preprocessing)
    protocol_path=folder/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()).get('preprocessing')!=spec:
        raise ValueError('Report preprocessing differs from the trained model protocol. Use the original run code/configuration.')
    history_plots(folder)
    result={}
    for split in ['val','test']:
        path=folder/f'{split}_predictions.csv'
        if not path.exists():raise FileNotFoundError(f'No completed predictions: {path}. Run training first.')
        frame=load_predictions(path,manifest,classes)
        directory=folder/'analysis'/split;directory.mkdir(parents=True,exist_ok=True)
        write_csv(frame,directory/'predictions.csv')
        errors=restore_review_fields(frame[~frame.correct].sort_values('confidence',ascending=False),directory/'misclassified_frames.csv')
        write_csv(errors,directory/'misclassified_frames.csv')
        report,pairs=confusion_outputs(frame,classes,directory)
        roc=roc_outputs(frame,classes,directory)
        video_summary(frame,classes,directory)
        result[split]={'frames':len(frame),'errors':int((~frame.correct).sum()),'accuracy':float(frame.correct.mean()),
                       'macro_f1':float(report['macro avg']['f1-score']),'weighted_f1':float(report['weighted avg']['f1-score']),
                       'roc':roc}
        if split=='test':
            copied=copy_test_frames(frame,classes,folder,options,preprocessing)
            expected=len(frame)
            for key,enabled in [('copied_original',options['test_originals']),('model_input_rgb',options['test_inputs'])]:
                if enabled and sum((folder/p).is_file() for p in copied[key])!=expected:
                    raise ValueError(f'Incomplete test-image export: {key}')
    save_json(folder/'analysis_summary.json',result)
    original=json.loads((folder/'metrics.json').read_text())
    row={k:original[k] for k in ['task','model','best_epoch','selection_metric','best_val_macro_f1','selected_val_loss','minimum_val_loss_seen','best_val_loss'] if k in original}
    for split in ['val','test']:
        row.update({split+'_'+k:v for k,v in result[split].items() if k!='roc'})
    write_csv(pd.DataFrame([row]),folder.parent/'summary.csv')
    save_json(folder/'analysis_complete.json',{'complete':True,'test_frames':result['test']['frames'],
        'test_errors':result['test']['errors'],'originals_copied':options['test_originals'],
        'model_input_rgb_copied':options['test_inputs'],'inference_repeated':False,
        'preprocessing_fingerprint':spec['fingerprint']})
    records=[{'relative_path':p.relative_to(folder).as_posix(),'bytes':p.stat().st_size,'extension':p.suffix}
             for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifacts.csv' and not p.name.endswith('.tmp')]
    write_csv(pd.DataFrame(records),folder/'artifacts.csv')
    print('Reports and test-frame copies saved:',folder,flush=True)
