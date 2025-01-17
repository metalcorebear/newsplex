# Newsplex

(C) 2025 Mark M. Bailey, PhD

## About

Newsplex uses topological data analysis to identify topological invariance in news discourse surrounding geopolitical events.<br>

The newsaggregator class uses the NewsAPI service to download news articles that bracket a specific date (date of event in question). It tokenizes text and preprocesses it for Word2Vec fitting and eventual topological analysis.<br>

The newsplex class fits the preprocessed data to a Word2Vec model, reduces its dimmensionality using t-distributed stochastic neighbor emebedding (tSNE), and finally computes topological features within a moving window (persistence distributions). It then calculates the Wasserstein distances between persistence distriibutions at each window. A large Wasserstein distance may indicate a topological shift in discourse - a potential signature of a significant event.

## References
@incollection{gudhi:WeightedRipsComplex
, author    = {Rapha{\"{e}}l Tinarrage and Yuichi Ike and Masatoshi Takenouchi}
, title     = {Weighted Rips Complex}
, publisher = {GUDHI Editorial Board}
, edition   = {3.10.1}
, booktitle = {GUDHI User and Reference Manual}
, url       = {https://gudhi.inria.fr/python/3.10.1/rips_complex_user.html#weighted-rips-complex}
, year      = {2024}
}

@incollection{gudhi:WassersteinDistance
, author    = {Th{\'{e}}o Lacombe and Marc Glisse}
, title     = {Wasserstein distance}
, publisher = {GUDHI Editorial Board}
, edition   = {3.10.1}
, booktitle = {GUDHI User and Reference Manual}
, url       = {https://gudhi.inria.fr/python/3.10.1/wasserstein_distance_user.html}
, year      = {2024}
}

@article{vanDerMaaten2008,
  author = {van der Maaten, L.J.P. and Hinton, G.E.},
  title = {Visualizing High-Dimensional Data Using t-SNE},
  journal = {Journal of Machine Learning Research},
  volume = {9},
  pages = {2579-2605},
  year = {2008}
}

@misc{vanDerMaatenTSNE,
  author = {van der Maaten, L.J.P.},
  title = {t-Distributed Stochastic Neighbor Embedding},
  howpublished = {\url{https://lvdmaaten.github.io/tsne/}}
}

@article{Belkina2019,
  author = {Belkina, A. C. and Ciccolella, C. O. and Anno, R. and Halpert, R. and Spidlen, J. and Snyder-Cappione, J. E.},
  title = {Automated optimized parameters for T-distributed stochastic neighbor embedding improve visualization and analysis of large datasets},
  journal = {Nature Communications},
  volume = {10},
  number = {1},
  pages = {1-12},
  year = {2019},
  doi = {10.1038/s41467-019-13055-y}
}
## Updates
* 2025-01-17: Initial commit.

## The Newsaggregator Class

* `# key_location should point to a JSON file containing the NewsAPI API key, with key value of 'key'`
* `# event_date is a date string of format 'YYYY-MM-DD'`
* `aggregator = newsplex.newsaggregator(key_location=key_location, filepath=output_path)`
* `aggregator.run_query(event_date)`
* `aggregator.generate_json()`

## The Newsplex Class

* `# Fits tSNE and calculates persistence distributions. Plots outputs.`
* `news_results = newsplex.newsplex(filepath=output_path)`
* `news_results.fit()`
* `news_results.plot_embeddings()`
* `news_results.calculate_persistences()`
