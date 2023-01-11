# Graphon
Project Repository for DD2430 in colloboration with [SEB](https://seb.se/).

# Project description 
In this project we intend to run some experiments to test the performance of <a href="https://en.wikipedia.org/wiki/Graphon">graphons</a> as embedding methods for downstream tasks (eg. classification, clustering) in relation to <a href="https://arxiv.org/pdf/1707.05005.pdf">Graph2Vec</a>, a common graph embedding framework.

<img src="https://user-images.githubusercontent.com/63954877/211856984-d14b76d5-05ee-4719-af5b-2d27cb161683.png" width=40%>
<figcaption><b>Fig: Graphon functions we used to generate data.</b></figcaption><br /> 

We ran our experiments on both Synthetic data (generated using the graphons) and Real data, using the framework <a href="https://wandb.ai/site">Weights & Biases</a> to efficiently store the results using the "Sweep" function.

Whaw we could find out is that graphons are impressively faster at finding an embedding compared to G2V, and the performance is always comparable if not higher in most cases.

As extra task we evaluated the use of graphons to perform augmentation on data. Useful in the case of unbalanced datasets.

<img src="https://user-images.githubusercontent.com/63954877/211858307-0a7fa982-448e-4081-afcc-0d455be8de2e.png" width=60%>
<figcaption><b>Fig: Idea behind data augmentation using graphons.</b></figcaption><br /> 

# How to use

To use the code it is necessary to set up the `config.yaml` file first, it contains descriptions of what each variable is used for; the `sweep_config.yaml` file has to be set up only if you intend to run a sweep over different parameters; in this case we have to set `SWEEP: True` in the previous file and set up the variables we want to sweep over following the same format as the ones present in the file.

After this the code can be run from the `main.py` function.

Note that if you intend to use the real data you have to uncomment the lines at the end of `main.py` and set `DOWNLOAD_DATA: True` in the first config file.


