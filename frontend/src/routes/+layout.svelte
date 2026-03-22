<script>
  let { children } = $props();
</script>

<svelte:head>
  <title>会計基準Q&A</title>
  <meta name="description" content="日本の会計基準に関するAI質問応答システム" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous" />
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&display=swap" rel="stylesheet" />
</svelte:head>

{@render children()}

<style>
  :global(*) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  :global(html) {
    height: 100%;
  }

  :global(body) {
    font-family: 'Noto Sans JP', sans-serif;
    background: #0c0c10;
    color: #e4e4e7;
    height: 100%;
    height: 100dvh;
    overflow: hidden;
  }
</style>
