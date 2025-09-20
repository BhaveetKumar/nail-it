# WASM App with Yew (Skeleton)

Prereqs: `wasm-pack` installed.

- Build:

```bash
wasm-pack build --target web
```

- Run (example):

```bash
python3 -m http.server 8080
```

Docs:

- [Yew Docs](https://yew.rs/docs/)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/)

Open `index.html` in a web server pointed at this directory after running `wasm-pack build --target web`. The generated JS is under `./pkg/`.
