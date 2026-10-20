import { defineConfig } from "vite";

export default defineConfig({
  base: "./", // rutas relativas: se puede servir desde cualquier subdirectorio (p. ej. GitHub Pages)
  build: { target: "es2022" },
  worker: { format: "es" },
  // ORT Web resuelve su .wasm con import.meta.url: no pre-empaquetarlo en dev
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
