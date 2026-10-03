const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <main style="font-family: system-ui; max-width: 720px; margin: 4rem auto">
    <h1>vocal-fast</h1>
    <p>实时变声器 UI 脚手架。音频设备和推理服务由 src-tauri 接入。</p>
    <p>当前基线：passthrough engine / VFRM model cache。</p>
  </main>`;
