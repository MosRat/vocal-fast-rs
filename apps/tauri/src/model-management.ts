import { escapeHtml as e, formatBytes, displayPath } from './format';
import { icon } from './icons';
import { on } from './view';
import { renderComponentBadges } from './model-components';
import type { GuiConfig, ModelCatalog } from './types';
import { importActive, observeImport, suggestIndex, type ModelMetadataView, type ModelMetadataEdit, type ModelProvider, type RemoteModelCatalog, type ModelImportStatus } from './model-library-types';

interface Dependencies {
  invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
  changed(): void;
  selected(path: string): void;
  completed(directory: string): Promise<void>;
  config(): GuiConfig;
  native(): boolean;
  completeBundle(): string | null;
}
const messages = (error: unknown): string => {
  if (error instanceof Error) return error.message;
  if (typeof error === 'object' && error) {
    const value = error as { message?: string; details?: unknown };
    if (value.message) return value.message;
    if (value.details) return messages(value.details);
  }
  return typeof error === 'string' ? error : JSON.stringify(error);
};

/** Model drafts and jobs have their own revisions; they never enter GuiConfig autosave. */
export class ModelManagement {
  private metadata: ModelMetadataView | null = null;
  private draft: ModelMetadataEdit | null = null;
  private metadataPath = '';
  private metadataRequest = 0;
  private remoteRequest = 0;
  private remote: RemoteModelCatalog | null = null;
  private pending = new Set<string>();
  private error: string | null = null;
  private notice: string | null = null;
  private completedJob = 0;
  private status: ModelImportStatus | null = null;
  private mode: 'remote' | 'local' = 'remote';
  private form = { provider: 'hfMirror' as ModelProvider, repository: '', revision: '', modelFile: '', indexFile: '', name: '', localModel: '', localIndex: '', directory: '', bundle: '', hubert: '', rmvpe: '' };
  constructor(private deps: Dependencies) {}

  async initialize(): Promise<void> {
    if (!this.deps.native()) return;
    try {
      const [directory, status] = await Promise.all([
        this.deps.invoke<string>('default_model_library'),
        this.deps.invoke<ModelImportStatus>('get_model_import_status'),
      ]);
      if (!this.form.directory) this.form.directory = directory;
      this.observe(status);
    } catch (error) { this.error = messages(error); }
    this.deps.changed();
  }

  observe(next: ModelImportStatus): void {
    const accepted = observeImport(this.status, next);
    if (accepted === this.status) return;
    this.status = accepted;
    if (next.phase === 'ready' && next.jobId > this.completedJob) {
      this.completedJob = next.jobId;
      const directory = next.libraryDirectory || this.form.directory || this.deps.config().modelDirectory;
      if (directory) void this.deps.completed(directory).catch(error => { this.error = messages(error); this.deps.changed(); });
    }
    this.deps.changed();
  }

  private async task(key: string, work: () => Promise<void>): Promise<void> {
    if (this.pending.has(key)) return;
    this.pending.add(key); this.error = null; this.notice = null; this.deps.changed();
    try { await work(); } catch (error) { this.error = messages(error); }
    finally { this.pending.delete(key); this.deps.changed(); }
  }
  private async edit(path: string): Promise<void> {
    const generation = ++this.metadataRequest;
    this.metadataPath = path; this.metadata = null; this.draft = null;
    await this.task('metadata', async () => {
      const view = await this.deps.invoke<ModelMetadataView>('read_model_metadata', { path });
      if (generation !== this.metadataRequest) return;
      this.metadata = view; this.draft = structuredClone(view.fields);
    });
  }
  private dirty(): boolean { return !!this.metadata && JSON.stringify(this.draft) !== JSON.stringify(this.metadata.fields); }
  private field(id: string, label: string, value: string, disabled = false, placeholder = ''): string {
    return `<label class="field-label"><span>${label}</span><input id="ml-${id}" value="${e(value)}" ${disabled ? 'disabled' : ''} placeholder="${e(placeholder)}"/></label>`;
  }
  render(catalog: ModelCatalog | null, locked: boolean): string {
    const native = this.deps.native();
    const active = !!this.status && importActive(this.status.phase);
    const importLocked = active || this.pending.has('start');
    const metadataLocked = locked || !native || this.pending.has('save');
    const selected = this.metadata?.inspection;
    const editor = this.draft ? `<div class="model-edit-grid">
      ${this.field('displayName', 'Display name', this.draft.displayName, metadataLocked)}
      ${this.field('author', 'Author', this.draft.author, metadataLocked)}
      ${this.field('language', 'Language', this.draft.language, metadataLocked, 'zh-CN')}
      ${this.field('license', 'License', this.draft.license, metadataLocked)}
      <div class="ml-wide">${this.field('source', 'Source / attribution', this.draft.source, metadataLocked)}</div>
      <div class="ml-wide">${this.field('tags', 'Tags (comma separated)', this.draft.tags.join(', '), metadataLocked)}</div>
      <label class="field-label ml-wide"><span>Notes</span><textarea id="ml-notes" rows="3" ${metadataLocked ? 'disabled' : ''}>${e(this.draft.notes ?? '')}</textarea></label>
      <div class="ml-wide ml-readonly"><span>${e(selected?.manifest.architecture ?? '')}</span><span>${selected?.manifest.sample_rate.toLocaleString()} Hz</span><span>${e(selected?.manifest.version ?? '')}</span><span>ID: ${e(selected?.manifest.id ?? '')}</span></div>
      <p class="ml-wide field-hint">Weights, architecture, speaker count and sample rate are read-only. Save updates display metadata while keeping the idle engine resident.</p>
      <div class="ml-wide ml-actions"><span class="ml-save-state" role="status">${this.notice ? icon('check', 14) + e(this.notice) : locked ? 'Stop audio before editing metadata' : this.dirty() ? 'Unsaved model details' : 'Model details are up to date'}</span><button id="ml-reload" class="button ghost" ${this.pending.has('metadata') || this.pending.has('save') ? 'disabled' : ''}>Reload details</button><button id="ml-save" class="button primary" ${metadataLocked || !this.metadata?.editable || !this.dirty() ? 'disabled' : ''}>${this.pending.has('save') ? 'Saving…' : 'Save model details'}</button></div>
    </div>` : `<div class="ml-empty">${icon('database', 28)}<strong>${this.pending.has('metadata') ? 'Reading model details…' : 'Select a VFRM to edit'}</strong><span>Names, notes, author, language, license and tags.</span></div>`;
    const options = (kind: 'models' | 'indices') => (this.remote?.files ?? []).filter(f => kind === 'models' ? f.kind !== 'index' : f.kind === 'index').map(f => `<option value="${e(f.path)}" ${(kind === 'models' ? this.form.modelFile : this.form.indexFile) === f.path ? 'selected' : ''}>${e(f.path)}${f.sizeBytes ? ` · ${formatBytes(f.sizeBytes)}` : ''}</option>`).join('');
    const acquisition = this.mode === 'remote' ? `<div class="model-edit-grid">
      <label class="field-label"><span>Source</span><select id="ml-provider" ${importLocked ? 'disabled' : ''}>${([['hfMirror','HF Mirror'],['huggingFace','Hugging Face'],['modelScope','ModelScope']] as const).map(([v,l])=>`<option value="${v}" ${v===this.form.provider?'selected':''}>${l}</option>`).join('')}</select></label>
      ${this.field('revision', 'Revision (optional)', this.form.revision, importLocked, this.form.provider === 'modelScope' ? 'master' : 'main')}
      <div class="ml-wide">${this.field('repository', 'Repository ID or URL', this.form.repository, importLocked, 'owner/model')}</div>
      <div class="ml-wide ml-example"><span>Chinese RVC example</span><button id="ml-example" class="button ghost compact" ${importLocked?'disabled':''}>maotianchen/maobailing_rvc</button></div>
      <div class="ml-wide ml-actions"><span class="field-hint">Public repositories. Choose only the model and matching index.</span><button id="ml-list" class="button secondary" ${importLocked || this.pending.has('list') || !native || !this.form.repository.trim() ? 'disabled' : ''}>${icon('refresh',14)}${this.pending.has('list') ? 'Reading files…' : 'Browse repository'}</button></div>
      ${this.remote ? `<label class="field-label ml-wide"><span>Voice weights / model archive</span><select id="ml-modelFile" ${importLocked ? 'disabled' : ''}><option value="">Choose a model file</option>${options('models')}</select></label><label class="field-label ml-wide"><span>Retrieval index</span><select id="ml-indexFile" ${importLocked || !this.form.modelFile.toLowerCase().endsWith('.pth') ? 'disabled' : ''}><option value="">No separate index</option>${options('indices')}</select></label><div class="ml-wide ml-revision">${this.remote.files.length} supported files · revision ${e(this.remote.repository.revision?.slice(0,12) ?? 'master')}</div>` : ''}
    </div>` : `<div class="model-edit-grid"><div class="ml-wide ml-path-picker">${this.field('localModel','RVC checkpoint, VFRM or ZIP',this.form.localModel,importLocked)}<button class="button secondary" data-ml-pick="localModel" ${!native||importLocked?'disabled':''}>${icon('folder',14)}Browse</button></div><div class="ml-wide ml-path-picker">${this.field('localIndex','Retrieval index (optional)',this.form.localIndex,importLocked)}<button class="button secondary" data-ml-pick="localIndex" ${!native||importLocked?'disabled':''}>Browse</button></div></div>`;
    return `<article class="panel model-management span-2" data-section-target="models">
      <div class="ml-header"><div><div class="kicker">MODEL WORKSPACE</div><h2>Your voice collection</h2><p>Manage details, download weights and build portable VFRM models.</p></div><span class="panel-badge">${icon('database',14)}${catalog?.entries.length ?? 0} models</span></div>
      ${this.error ? `<div class="ml-error" role="alert">${icon('alert',16)}<span>${e(this.error)}</span><button id="ml-dismiss" class="button ghost compact" aria-label="Dismiss model error">Dismiss</button></div>` : ''}
      ${this.notice ? `<div class="ml-notice" role="status">${icon('check',14)}${e(this.notice)}</div>` : ''}
      <div class="ml-grid"><section class="ml-card"><h3>Model details</h3><label class="field-label"><span>Library model</span><select id="ml-edit-path" ${this.pending.has('metadata') || this.pending.has('save') ? 'disabled' : ''}><option value="">Choose a VFRM</option>${(catalog?.entries ?? []).filter(f=>f.kind==='vfrm').map(f=>`<option value="${e(f.path)}" ${f.path===this.metadataPath?'selected':''}>${e(f.displayName)}</option>`).join('')}</select></label>${this.metadata ? `<div class="ml-selected-path" title="${e(this.metadata.inspection.path)}">${e(displayPath(this.metadata.inspection.path))}</div>${renderComponentBadges(this.metadata.inspection.components)}` : ''}${editor}</section>
      <section class="ml-card"><h3>Add a model</h3><div class="ml-mode" role="group" aria-label="Import source"><button data-ml-mode="remote" class="button ${this.mode==='remote'?'secondary active':'ghost'}" ${importLocked?'disabled':''}>${icon('download',14)}Download</button><button data-ml-mode="local" class="button ${this.mode==='local'?'secondary active':'ghost'}" ${importLocked?'disabled':''}>${icon('folder',14)}Local files</button></div>
      ${acquisition}<div class="model-edit-grid ml-import-options">${this.field('name','New model name',this.form.name,importLocked)}${this.field('directory','Destination library',this.form.directory,importLocked)}
      <details class="ml-wide ml-acoustic"><summary>Shared acoustic weights</summary><p class="field-hint">Automatically reuse a complete model in the destination library. You can also select a complete VFRM or two canonical acoustic VFRM files. Raw HuBERT / RMVPE PT and ONNX exports are not converted here.</p><div class="ml-path-picker">${this.field('bundle','Complete model (optional)',this.form.bundle,importLocked)}<button class="button ghost" data-ml-pick="bundle" ${!native||importLocked?'disabled':''}>Browse</button></div>${this.field('hubert','HuBERT canonical VFRM',this.form.hubert,importLocked)}${this.field('rmvpe','RMVPE canonical VFRM',this.form.rmvpe,importLocked)}</details></div>
      <div class="ml-actions"><span class="field-hint">${this.mode==='remote'?'Downloads are cached and verified.':'Original files are preserved.'}</span><button id="ml-start" class="button primary" ${!native||importLocked||!this.form.name.trim()||!(this.mode==='remote'?this.remote&&this.form.modelFile:this.form.localModel)?'disabled':''}>${icon('download',14)}${this.pending.has('start')?'Starting…':this.mode==='remote'?'Download & convert':'Import & convert'}</button></div>
      ${this.renderProgress()}
      </section></div></article>`;
  }

  private renderProgress(): string {
    const s=this.status;if(!s||s.phase==='idle')return '';
    const active=importActive(s.phase);const percent=s.totalBytes&&s.phase==='downloading'?Math.min(100,s.downloadedBytes/s.totalBytes*100):null;
    return `<section class="ml-progress ${s.phase==='failed'?'failed':s.phase==='ready'?'ready':''}" aria-label="Model import progress"><div class="ml-progress-heading"><strong>${e(s.message)}</strong><span>${(s.elapsedMs/1000).toFixed(1)} s</span></div>${active?`<progress ${percent===null?'':`value="${percent}" max="100"`} aria-label="${e(s.phase)} progress"></progress>`:''}${s.file?`<div class="ml-selected-path">${e(displayPath(s.file))}</div>`:''}${s.phase==='downloading'?`<small>${formatBytes(s.downloadedBytes)}${s.totalBytes?` / ${formatBytes(s.totalBytes)}`:''}</small>`:''}${s.warnings.map(w=>`<p class="ml-warning">${icon('alert',14)}${e(w)}</p>`).join('')}<div class="ml-actions"><span class="tag">${s.phase==='ready'?(s.complete?'Complete VFRM':'Synth-only VFRM'):e(s.phase)}</span>${active?`<button id="ml-cancel" class="button ghost" ${s.phase==='publishing'||s.phase==='cancelling'?'disabled':''}>Cancel import</button>`:s.phase==='ready'&&s.outputPath?`<button id="ml-use" class="button secondary">Use in Studio</button>`:s.phase==='failed'||s.phase==='cancelled'?`<button id="ml-retry" class="button secondary">Retry import</button>`:''}</div></section>`;
  }

  wire(locked: boolean): void {
    const byId=<T extends HTMLElement>(id:string)=>document.getElementById(`ml-${id}`) as T|null;
    on(byId<HTMLSelectElement>('edit-path'),'change',event=>{const path=(event.target as HTMLSelectElement).value;if(path)void this.edit(path);});
    for(const key of ['displayName','source','author','language','license','tags','notes'] as const){
      on(byId<HTMLInputElement>(key),'input',event=>{
        if(!this.draft||locked||this.pending.has('save'))return;
        const value=(event.target as HTMLInputElement).value;
        if(key==='tags')this.draft.tags=value.split(',').map(s=>s.trim()).filter(Boolean);
        else if(key==='notes')this.draft.notes=value||null;
        else this.draft[key]=value;
        const save=byId<HTMLButtonElement>('save');if(save)save.disabled=!this.dirty()||!this.metadata?.editable||!this.deps.native();
      });
    }
    on(byId('reload'),'click',()=>{if(this.metadataPath)void this.edit(this.metadataPath);});
    on(byId('save'),'click',()=>{if(!this.metadata||!this.draft||locked)return;const path=this.metadata.inspection.path;const revision=this.metadata.revision;const edit=structuredClone(this.draft);void this.task('save',async()=>{const view=await this.deps.invoke<ModelMetadataView>('edit_model_metadata',{path,revision,edit});this.metadata=view;this.draft=structuredClone(view.fields);this.notice='Model details saved';await this.deps.completed(this.deps.config().modelDirectory??this.form.directory);});});
    for(const key of Object.keys(this.form) as (keyof typeof this.form)[]){
      on(byId<HTMLInputElement>(key),'input',event=>{
        if(this.status&&importActive(this.status.phase))return;
        const value=(event.target as HTMLInputElement).value;
        if(key==='provider')this.form.provider=value as ModelProvider;else this.form[key]=value;
        if(['provider','repository','revision'].includes(key)){this.remoteRequest++;this.remote=null;this.form.modelFile='';this.form.indexFile='';}
        if(key==='modelFile'){
          this.form.indexFile=value.toLowerCase().endsWith('.pth')?suggestIndex(value,this.remote?.files??[]):'';
          if(!this.form.name)this.form.name=value.split('/').pop()?.replace(/\.[^.]+$/,'')??'';
          this.deps.changed();
        }else if(key==='provider')this.deps.changed();
        const list=byId<HTMLButtonElement>('list');if(list)list.disabled=!this.form.repository.trim()||this.pending.has('list')||!this.deps.native();
        const start=byId<HTMLButtonElement>('start');if(start)start.disabled=!this.deps.native()||!this.form.name.trim()||!(this.mode==='remote'?this.remote&&this.form.modelFile:this.form.localModel)||this.pending.has('start');
      });
    }
    document.querySelectorAll<HTMLElement>('[data-ml-mode]').forEach(el=>on(el,'click',()=>{this.mode=el.dataset.mlMode as 'remote'|'local';this.deps.changed();}));
    document.querySelectorAll<HTMLElement>('[data-ml-pick]').forEach(el=>on(el,'click',()=>{const key=el.dataset.mlPick as 'localModel'|'localIndex'|'bundle';void this.task('pick',async()=>{const path=await this.deps.invoke<string|null>('pick_model_file',{kind:key==='localIndex'?'index':key==='bundle'?'acoustic':'model'});if(path){this.form[key]=path;if(key==='localModel'&&!this.form.name)this.form.name=path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/,'')??'';}});}));
    on(byId('list'),'click',()=>{const generation=++this.remoteRequest;const repository={provider:this.form.provider,repository:this.form.repository,revision:this.form.revision||null};void this.task('list',async()=>{const result=await this.deps.invoke<RemoteModelCatalog>('list_remote_models',{repository});if(generation!==this.remoteRequest)return;this.remote=result;this.form.modelFile='';this.form.indexFile='';if(!result.files.length)this.notice='No supported RVC checkpoints, VFRM files or ZIP archives found.';});});
    on(byId('example'),'click',()=>{this.remoteRequest++;this.remote=null;this.form.provider='huggingFace';this.form.repository='maotianchen/maobailing_rvc';this.form.revision='';this.form.modelFile='';this.form.indexFile='';this.deps.changed();});
    const start=()=>void this.task('start',async()=>{
      const source=this.mode==='remote'?{kind:'remote',repository:this.remote?.repository,modelFile:this.form.modelFile,indexFile:this.form.indexFile||null}:{kind:'local',modelPath:this.form.localModel,indexPath:this.form.localIndex||null};
      const request={source,displayName:this.form.name,libraryDirectory:this.form.directory||null,acousticBundle:this.form.bundle||(!this.form.hubert&&!this.form.rmvpe?this.deps.completeBundle():null),hubertSource:this.form.hubert||null,rmvpeSource:this.form.rmvpe||null};
      this.observe(await this.deps.invoke<ModelImportStatus>('start_model_import',{request}));
    });
    on(byId('start'),'click',start);on(byId('retry'),'click',start);
    on(byId('cancel'),'click',()=>void this.task('cancel',async()=>{this.observe(await this.deps.invoke<ModelImportStatus>('cancel_model_import',{jobId:this.status?.jobId}));}));
    on(byId('use'),'click',()=>{if(this.status?.outputPath)this.deps.selected(this.status.outputPath);});
    on(byId('dismiss'),'click',()=>{this.error=null;this.deps.changed();});
  }
}
