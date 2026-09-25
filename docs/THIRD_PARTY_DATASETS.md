# Third-party datasets

Raw videos, original annotations, preprocessed HDF5 files, and third-party model weights are deliberately excluded. Obtain them from the authorised sources and comply with their licences and access terms.

| Dataset | Version/records used | Official source | Primary citation | Redistribution note |
|---|---|---|---|---|
| QFVS (Query-Focused Video Summarization) | CVPR 2017 benchmark release; videos `P01`–`P04`; all 45 concept-pair queries per video represented in `qfvs_query_index.csv` | Paper-advertised project page: http://www.aidean-sharghi.com/cvpr2017 ; archival paper page: https://openaccess.thecvf.com/content_cvpr_2017/html/Sharghi_Query-Focused_Video_Summarization_CVPR_2017_paper.html | Sharghi, A., Laurel, J. S. & Gong, B. Query-Focused Video Summarization: Dataset, Evaluation, and a Memory Network Based Approach. CVPR, 4788–4797 (2017). | The package provides identifiers, concept mappings, selected shot indices, and derived scores only. Original videos and annotations are not redistributed. |
| TVSum | TVSum50 / Yahoo Webscope version 1.1 annotation convention; 50 records (`video_1`–`video_50`) in the standard-protocol table | https://github.com/yalesong/tvsum and http://people.csail.mit.edu/yalesong/tvsum | Song, Y., Vallmitjana, J., Stent, A. & Jaimes, A. TVSum: Summarizing Web Videos Using Titles. CVPR, 5179–5187 (2015). | The official repository states that videos were collected from YouTube under CC BY 3.0. This release nevertheless avoids repackaging the videos and benchmark annotations. |
| SumMe | Original 25-video benchmark; records `video_1`–`video_25` in the standard-protocol table | https://gyglim.github.io/me/vsum/index.html#benchmark | Gygli, M., Grabner, H., Riemenschneider, H. & Van Gool, L. Creating Summaries from User Videos. ECCV, 505–520 (2014). | Only video identifiers and derived evaluation results are included; source videos and annotations remain with the original distributor. |

## Local layout for full reruns

Place authorised QFVS material under `code/research_q3/data/qfvs/` with the original annotation layout expected by the scripts. Place TVSum and SumMe HDF5 conversions at paths supplied to the standard-protocol script. Preprocessing entry points are:

```bash
cd code
python -m research_q3.experiments.cache_qfvs_features --help
python -m research_q3.experiments.cache_qfvs_openai_clip --help
python -m research_q3.experiments.run_standard_protocol --help
```

The repository does not provide a downloader because automated redistribution or scraping could conflict with source terms.
