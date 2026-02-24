# Тюнинг полноты и sliced inference (StarDist)

Новые CLI-скрипты:
- `nuclei_cli.py` — инференс, пресеты, диагностика, grid-search.
- `smoke_infer.py` — минимальный smoke для обычного инференса.
- `smoke_tiled_roi.py` — smoke для sliced inference + ROI (есть `--mock` режим).

## Пресеты

Доступны runtime-пресеты:
- `legacy` — прежние значения проекта (поведение сохранено).
- `balanced` — умеренный пресет, мягкий purple filter.
- `high_recall` — пресет на полноту (минимум отсева).

## 1) Legacy запуск

```bash
python nuclei_cli.py infer \
  --image "Гематоксилин-эозин сетчатка без воздействия/Snap-44400.jpg" \
  --preset legacy \
  --detector-backend stardist
```

## 2) High Recall запуск

```bash
python nuclei_cli.py infer \
  --image "Гематоксилин-эозин сетчатка без воздействия/Snap-44400.jpg" \
  --preset high_recall \
  --detector-backend stardist
```

## 3) Диагностика фильтров

```bash
python nuclei_cli.py infer \
  --image "Гематоксилин-эозин сетчатка без воздействия/Snap-44400.jpg" \
  --preset balanced \
  --detector-backend stardist \
  --diagnostics
```

Альтернатива: `--debug-filters` (синоним).

Вывод включает:
- кандидаты до NMS (если модель отдает это число);
- после NMS;
- отсев по `min_area`;
- отсев purple filter по причинам (`center`, `ratio`, `other`) или `weak_color` в мягком режиме;
- финальное число;
- статистики площадей.

## 4) Sliced inference с калибровкой средней клетки

### Вариант A: задать диаметр клетки

```bash
python nuclei_cli.py infer \
  --image "Гематоксилин-эозин сетчатка без воздействия/Snap-44400.jpg" \
  --preset balanced \
  --detector-backend stardist \
  --sliced-inference \
  --average-cell-diameter-px 12 \
  --tile-factor 64 \
  --overlap-ratio 0.25 \
  --tile-min-px 512 \
  --tile-max-px 2048 \
  --merge-iou-thresh 0.3 \
  --border-trim-enabled \
  --border-trim-factor 1.0 \
  --diagnostics
```

### Вариант B: задать bbox средней клетки

```bash
python nuclei_cli.py infer \
  --image "Гематоксилин-эозин сетчатка без воздействия/Snap-44400.jpg" \
  --preset balanced \
  --detector-backend stardist \
  --sliced-inference \
  --average-cell-bbox 120,180,14,16
```

## 5) ROI в CLI

### Прямоугольники (можно несколько)

```bash
python nuclei_cli.py infer \
  --image ".../Snap-44400.jpg" \
  --sliced-inference \
  --average-cell-diameter-px 12 \
  --roi-rect 100,100,600,500 \
  --roi-rect 900,120,400,350
```

### Полигоны (можно несколько)

```bash
python nuclei_cli.py infer \
  --image ".../Snap-44400.jpg" \
  --sliced-inference \
  --average-cell-diameter-px 12 \
  --roi-poly 100,100,700,120,650,600,140,550
```

### ROI из JSON/GeoJSON

```bash
python nuclei_cli.py infer \
  --image ".../Snap-44400.jpg" \
  --sliced-inference \
  --average-cell-diameter-px 12 \
  --roi-file roi.json
```

Поддерживаются форматы:
- GeoJSON `Polygon`/`MultiPolygon`.
- Простой JSON:
  - `{"polygon": [[x,y], ...]}`
  - `{"rect": [x,y,w,h]}`
  - множественные: `{"polygons": [...], "rects": [...]}`.

## 6) Конфиг JSON (пример)

```json
{
  "preset": "balanced",
  "detector_backend": "stardist",
  "sliced_inference": true,
  "average_cell_diameter_px": 12.0,
  "tile_factor": 64,
  "overlap_ratio": 0.25,
  "tile_min_px": 512,
  "tile_max_px": 2048,
  "min_roi_cover_ratio": 0.05,
  "merge_iou_thresh": 0.3,
  "border_trim_enabled": true,
  "border_trim_factor": 1.0,
  "roi_rect": [[100, 100, 600, 500]],
  "roi_poly": [[[900, 120], [1300, 120], [1280, 500], [930, 480]]]
}
```

Запуск:

```bash
python nuclei_cli.py infer --image ".../Snap-44400.jpg" --config tuned_config.json --diagnostics
```

## 7) Тюнинг без разметки

```bash
python nuclei_cli.py tune \
  --images-dir "Гематоксилин-эозин сетчатка без воздействия" \
  --preset balanced \
  --detector-backend stardist \
  --num-images 5 \
  --output-config tuned_config.json
```

Сетка по умолчанию:
- `prob_thresh`: `0.05, 0.1, 0.15`
- `nms_thresh`: `0.3, 0.4, 0.55`
- `min_area_px`: `0, 10, 18`
- `scale`: `0.75, 1.0, 1.25`
- `purple_filter_enabled`: `0, 1`
- `require_center_purple`: `0, 1`
- `min_purple_ratio`: `0.05, 0.1, 0.2`
- `purple_s_min`: `15, 30, 45`
- `purple_v_max`: `230, 255`

Top-10 сортируются по `avg_objects_per_image` со штрафом при `small_objects_share > 0.35`.

## 8) Сохранение результатов

```bash
python nuclei_cli.py infer \
  --image ".../Snap-44400.jpg" \
  --preset balanced \
  --output-json detections.json \
  --output-overlay detections_overlay.png
```

## 9) Smoke-проверки

Обычный smoke:

```bash
python smoke_infer.py --image ".../Snap-44400.jpg" --preset legacy --detector-backend stardist
```

Sliced + ROI smoke (без зависимости от StarDist/Cellpose в окружении):

```bash
python smoke_tiled_roi.py --mock --image ".../Snap-44400.jpg"
```

Этот smoke проверяет:
- запуск sliced pipeline без ROI;
- запуск с ROI;
- отсутствие центров финальных объектов вне ROI.
