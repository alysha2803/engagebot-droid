# Models

Not committed to git (binary, 13MB). `face_embedding.py` needs `w600k_mbf.onnx`
here -- the ArcFace recognition model from InsightFace's official `buffalo_sc`
release pack.

```bash
curl -sL "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip" -o buffalo_sc.zip
unzip -o -d buffalo_sc buffalo_sc.zip
mv buffalo_sc/w600k_mbf.onnx w600k_mbf.onnx
rm -rf buffalo_sc buffalo_sc.zip
```

(The pack also includes `det_500m.onnx`, a face *detector* -- unused here
since MediaPipe already handles detection. Safe to delete.)

Source: https://github.com/deepinsight/insightface (non-commercial research
license per their model zoo README -- fine for this FYP, revisit if this
ever goes beyond academic use).
