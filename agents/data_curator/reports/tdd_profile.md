# TDD Dataset Profile

- Dataset root: `/data/data2/yiyang/DentalClaw/data/TDD`
- Total cases: 1000
- Fully matched cases: 1000

## Counts

- Radiographs: 1000
- Teeth masks: 1000
- Maxillomandibular masks: 1000
- BBox entries: 1000

## Supported Export Targets

- Detection: COCO-style teeth detection from teeth_bbox.json
- Teeth segmentation: Binary PNG masks thresholded from Segmentation/teeth_mask
- Maxillomandibular segmentation: Binary PNG masks thresholded from Segmentation/maxillomandibular

## Notes

- Teeth mask mode: RGB
- Maxillomandibular mask mode: L
- Export strategy: JPEG masks are thresholded to binary PNG during export (default threshold=127).

## BBox Stats

- Min boxes/case: 0
- Max boxes/case: 48
- Avg boxes/case: 26.00
- Non-numeric labels: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T
