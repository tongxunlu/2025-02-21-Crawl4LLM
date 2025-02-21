当传统爬虫还在比拼抓取速度，AI训练早已进入"数据质量战争"时代。传统的网页爬虫工具虽然能够抓取大量信息，但效率和质量仍然是瓶颈。不仅导致了数据浪费，还增加了计算资源的消耗。今天刚好可以分享一款智能的爬虫系统：Crawl4LLM，正是为了解决这些问题而诞生的。这款系统通过智能评估网页对LLM预训练的影响力，能够在更短的时间内抓取更有价值的网页，提升预训练效率，减少不必要的数据抓取，带来了近5倍的效率提升。项目简介Crawl4LLM 是清华大学和卡内基梅隆大学联合开源的一个智能爬虫系统。专为提升 LLM 预训练效率而设计。它的核心优势在于智能评估网页对LLM预训练的影响力，并根据网页的预期价值优先抓取有意义的数据。相比于传统爬虫需要抓取100个网页才能获得所需的效果，Crawl4LLM只需抓取21个网页就能达到同样的效果，效率提升了近5倍！这不仅节省了大量的计算资源，还能够显著提高爬取数据的质量。主要功能与亮点1、智能化网页选择Crawl4LLM 通过智能评估哪些网页对 LLM 的预训练更有价值，基于这一评估结果，优先选择抓取高价值网页，保证训练数据的质量和模型效果。2、三种灵活的爬取模式提供了三种不同的爬取模式，用户可以根据不同的需求选择：• Crawl4LLM智能模式：该模式是 Crawl4LLM 的核心，能够智能选择最具价值的网页进行抓取，最大化抓取效率。• 随机爬取模式：适用于那些不需要精确选择网页内容的情况，像传统爬虫一样随机抓取网页。• 基于链接数量的爬取模式：通过网页上的链接数量来决定爬取的网页，适用于需要大规模数据抓取的场景。3、爬虫状态定期保存系统支持定期保存爬虫状态，确保即使在出现中断时，也能从中断点继续抓取，避免数据丢失。4、集成数据浏览工具Crawl4LLM 提供了数据浏览工具，可以帮助用户更方便地查看抓取的数据，并对数据进行分析和处理。通过直观的可视化界面，用户可以实时监控爬虫抓取的进度和效果。5、完整工具链与DCLM框架对接Crawl4LLM 不仅支持网页抓取，还可以提取文档ID、获取文档内容，并能与DCLM（Deep Learning Model）预训练框架无缝对接。这意味着，爬取的数据可以直接用于模型的训练，提高数据流的效率和准确性。快速使用Crawl4LLM 系统是由 Python 语言100%开发完成，所以只需要本地有相关的 Python 环境即可使用。必要准备：clueweb22数据集、Python 3.10及以上、DCLM fastText一切准备就绪，执行下面的命令即可：python crawl.py crawl --config <path_to_your_config_file>详细的配置和参数说明，可前往项目主页查看。适用场景• 大规模LLM预训练• 数据集构建• 搜索引擎优化• 网络监测与分析写在最后Crawl4LLM 通过智能评估网页对 LLM 预训练的影响力，提升了数据抓取的效率和质量。通过灵活的爬取模式、数据可视化工具以及完整的工具链，它极大简化了爬虫系统的部署与使用，为从事LLM预训练的团队提供了一种更加高效、精准的解决方案。相比传统的爬虫系统，它不仅提高了抓取效率，还大幅提升了数据质量，为各类数据分析和AI训练任务提供了更为高效和智能的支持。GitHub 项目地址：https://github.com/cxcscmu/Crawl4LLM

# Crawl4LLM

This repo contains the code for the paper "[Crawl4LLM: Efficient Web Crawling for LLM Pretraining](https://arxiv.org/pdf/2502.13347)".

## Prerequisite

1. [Request the ClueWeb22 dataset](https://lemurproject.org/clueweb22/).
2. Create a virtual environment with python >= 3.10 and install the following requirements:
```
numpy
tqdm
fasttext
pyyaml
wandb
```
3. [Download the DCLM fastText classifier](https://huggingface.co/mlfoundations/fasttext-oh-eli5/tree/main) to `fasttext_scorers/`.

> [!IMPORTANT] 
> To run the crawler efficiently, the ClueWeb22 data should be placed on **an SSD**.

## Run the Crawler

To run a (simulated) crawl, first create a yaml configuration file under `configs/`, and run the following command:

```bash
python crawl.py crawl --config <path_to_your_config_file>
```

### Crawl4LLM

Create a yaml file in `configs/` with the following content:

```yaml
cw22_root_path: <path_to_clueweb22_a>
seed_docs_file: seed.txt
output_dir: crawl_results/seed_10k_crawl_20m_dclm_fasttext
num_selected_docs_per_iter: 10000
num_workers: 16  # set to a number that fits your machine
save_state_every: -1  # set to a positive number to save the state (queue & visited set) of the crawler every certain steps
max_num_docs: 20000000
selection_method: dclm_fasttext_score
order: desc  # desc for descending, asc for ascending
wandb: true  # set to false to disable wandb logging
wandb_project: crawler
wandb_run_name: seed_10k_crawl_20m_dclm_fasttext
rating_methods:
    - 
        type: length
    - 
        type: fasttext_score
        rater_name: dclm_fasttext_score
        model_path: fasttext_scorers/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin
```

Documents are scored by all scorers in `rating_methods`. In the above configuration file, we set a `length` scorer, which scores a document by its length, and a `fasttext_score` scorer which uses the DCLM fastText model to score a document. The final ranking is determined by `selection_method` which is set to `dclm_fasttext_score`, the name of the `fasttext_score` scorer.

### Baseline Crawlers

#### Random Crawler

```yaml
cw22_root_path: <path_to_clueweb22_a>
seed_docs_file: seed.txt
output_dir: crawl_results/seed_10k_crawl_20m_random
num_selected_docs_per_iter: 10000
num_workers: 16
save_state_every: -1
max_num_docs: 20000000
selection_method: random_score
order: desc
wandb: true
wandb_project: crawler
wandb_run_name: seed_10k_crawl_20m_random
rating_methods:
    - 
        type: random_score
```

#### Indegree-based Crawler

```yaml
cw22_root_path: <path_to_clueweb22_a>
seed_docs_file: seed.txt
output_dir: crawl_results/seed_10k_crawl_20m_indegree
num_selected_docs_per_iter: 10000
num_workers: 16
save_state_every: -1
max_num_docs: 20000000
selection_method: inlink_count
order: desc
wandb: true
wandb_project: crawler
wandb_run_name: seed_10k_crawl_20m_indegree
rating_methods:
    - 
        type: inlink_count
```

## Pretraining and Evaluation

After running the crawler, the crawled document ids will be placed in `output_dir` in the configuration file. Run the following command to get the document texts:

```bash
python fetch_docs.py  --input_dir <document_ids_dir>  --output_dir <document_texts_dir>  --num_workers <num_workers>
```

Then you can use the [DCLM](https://github.com/mlfoundations/dclm/) framework to run LLM pretraining and evaluation.

## Miscellaneous

### Browse the Data

Run the following command to print a document and its outlinks by its id:

```bash
python access_data.py <path_to_clueweb22> <document_id>
```
