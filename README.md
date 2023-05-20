# 🚀 A-Unified-Framework-for-Deep-Attribute-Graph-Clustering

☞ See the Chinese version in \[[Marigold](https://www.marigold.website/readArticle?workId=145&author=Marigold&authorId=1000001)\]

Recently, attribute graph clustering has developed rapidly, at the same time various deep attribute graph clustering methods have sprung up. Although most of the methods are open source, it is a pity that these codes do not have a unified framework, which makes researchers have to spend a lot of time modifying the code to achieve the purpose of reproduction. Fortunately, Liu et al. \[Homepage: [yueliu1999](https://github.com/yueliu1999)\] organized the deep graph clustering method into a code warehouse—— [ Awesome-Deep-Graph-Clustering(ADGC)](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering). For example, they provided more than 20 datasets and unified the format. Moreover, they list the most related paper about deep graph clustering  and give the link of source code. It is worth mentioning that they organize the code of deep graph clustering into rand-augmentation-model-clustering-visualization-utils structure, which greatly facilitates beginners and researchers. Here, on behalf of myself, I would like to express my sincere thanks and high respect to Liu et al. 

❤️ **Acknowledgements:**

Thanks for the open source of these authors (not listed in order):

\[ [yueliu1999](https://github.com/yueliu1999) | [bdy9527](https://github.com/bdy9527)| [Liam Liu](https://github.com/Tiger101010) | [Zhihao PENG](https://github.com/ZhihaoPENG-CityU) | [William Zhu](https://github.com/grcai)\]

<a href="https://github.com/yueliu1999" target="_blank"><img src="https://avatars.githubusercontent.com/u/41297969?s=64&v=4" alt="yueliu1999" width="48" height="48"/> </a> <a href="https://github.com/bdy9527" target="_blank"><img src="https://avatars.githubusercontent.com/u/16743085?s=64&v=4" alt="bdy9527" width="48" height="48"/></a> <a href="https://github.com/Tiger101010" target="_blank"><img src="https://avatars.githubusercontent.com/u/34651180?s=64&v=4" alt="Liam Liu" width="48" height="48"/> </a> <a href="https://github.com/ZhihaoPENG-CityU" target="_blank"><img src="https://avatars.githubusercontent.com/u/23076563?s=64&v=4" alt="Zhihao PENG" width="48" height="48"/> </a> <a href="https://github.com/grcai" target="_blank"><img src="https://avatars.githubusercontent.com/u/38714987?s=64&v=4" alt="William Zhu" width="48" height="48"/></a>

## 🍉 Introduction

On the basis of ADGC, I refactored the code to make the deep clustering code achieve a higher level of unification. Specifically, I redesigned the architecture of the code, so that you can run the open source code easily. I defined some tool classes and functions to simplify the code and make the settings' configuration clear.  

- 📃 `main.py`: The **entrance** file of my framework.
- 📃 `requirements.txt`: The third-party library environments that need to be installed first.
- 📁 `dataset`: The directory including the dataset you need, whose subdirectories are named after dataset names. The subdirectory includes the features file, the labels file and the adjacency matrix file, named after **{dataset name}\_feat.npy**, **{dataset name}\_label.npy** and **{dataset name}\_adj.npy**, such as **acm\_feat.npy**, **acm\_label.npy** and **acm\_adj.npy**. **Besides**, the dataset directory also includes a python file named dataset\_info.py which stores the information related to datasets.
- 📁 `module`: The directory including the most used basic modules of model, such as the Auto-encoder (**AE.py**), the Graph Convolutional Layer (**GCN.py**), the Graph Attention Layer (**GAT.py**), et al.
- 📁 `model`: The directory including the model you want to run. The directory format is a subdirectory named after the uppercase letters of the model name, which contains two files, one is the model file **model.py** for storing model classes, and the other is the training file **train.py** for model training. Our framework will dynamically import the training file of the model according to the input model name. **Besides**, it can also store the pre-training directory named the lowercase letters of pretrain\_{module name}\_for\_{model name}, which stores the **train.py** file. For example, if you want to pretrain the AE module in SDCN, you can named the directory with **pretrain\_ae\_for\_sdcn**. The model.py file and train.py file can be overwritten according to the template provided in the template directory. The explanation.txt file provides the attributes that argparse has, and you can use them according to your needs.
- 🛠️`utils`: The directory including some **tool** classes and functions.
  - 💾 `load_data.py`: It includes the functions of  **loading dataset** for training.
  - 📊 `data_processor.py`: It includes the functions of transferring data storing types and others, such as **numpy to torch**, **symmetric normalization** et al.
  - ~~`calculator.py`:~~ ~~It includes the function of calculating **mean** and standard difference.~~ This file has been merged into `utils.py`.
  - 📊 `evalution.py`: It includes the function of calculating the related **metrics** of clustering, such as ACC, NMI, ARI and F1\_score.
  - ~~`formatter.py`: It includes the function of **formatting** the output of **variables** according to your input variables.~~ This file has been merged into `utils.py`.
  - 📃 `logger.py`: It includes **a log class**, through which you can record the information you want to output to the log file.
  - ~~`parameter_counter.py`: It includes the function of counting the model's parameters.~~ This file has been merged into `utils.py`.
  - 📁 `path_manager.py`: It includes the function of  transforming **the relative path** to **the absolute path** if needed. Of course, if you don't need, it also should be called because it also **stores the path** needed by the training, such as the storing path of logs, pretrain parameters files, clustering visualization images, et al.
  - 🎨 `plot.py`: It includes the function of drawing clustering visualization via **TSNE** and save the image. The features heatmap will also be developed soon later.
  - ⏱️`time_manager.py`: It includes **a time class** to record time consuming and a function to format datetime.
  - 🎲 `rand.py`: It includes the function of set random seed.
  - 🛠️ `utils.py`: It includes the tools function from pervious file, such as `get_format_variables()` from `fomatter.py`.
  - ⚙️ `options.py`: It includes the argparse object.
- 📁 `logs`: The directory is used to **store the output logs files**. Its subdirectories are named after the model names and the logs files are named after the start time.
- 📁 `pretrain`:  The directory is used to **store the pre-training parameters files**. Its subdirectories are named after the format of pretrain\_{module name}. Parameters files are categorized by model and dataset name.
- 🖼️`img`: The directory is used to **store the output images**, whose subdirectories are named after **clustering** and **heatmap**.

## 🍓 Quick Start

After git clone the code, you can follow the steps below to run:

✈️ `Step 1`: Check the environment or run the requirements.txt to install the libraries directly.

```bash
pip install -r requirements.txt
```

✈️ `Step 2`: Prepare the datasets. If you don't have the datasets, you can download them from Liu's warehouse \[[yueliu1999](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering) | [Google Drive](https://drive.google.com/drive/folders/1thSxtAexbvOyjx-bJre8D4OyFKsBe1bK) | [Nutstore](https://www.jianguoyun.com/p/DfzK1pwQwdaSChjI2aME)\]. Then unzip them to the dataset directory.

✈️ `Step 3`: Run the file in the directory where main.py is located in command line. If it is in the integrated compilation environment, you can directly run the main.py file. 

### :star: Examples

#### Example 1

 **Take the training of the DAEGC as example:**

:one: **pretrain GAT:**

```shell
python main.py --pretrain --model pretrain_gat_for_daegc --dataset acm  --t 2 --desc pretrain_the_GAT_for_DAEGC_on_acm
# or the simplified command:
python main.py -P -M pretrain_gat_for_daegc -D acm -T 2 -DS pretrain_the_GAT_for_DAEGC_on_acm
```

:two: **train DAEGC:**

```shell
python main.py --model DAEGC --dataset cora --t 2 -desc Train_DAEGC_1_iteration_on_the_ACM_dataset
# or the simplified command:
python main.py -M DAEGC -D cora -T 2 -DS Train_DAEGC_1_iteration_on_the_ACM_dataset
```

####  Example 2

**Take the training of the SDCN as example:**

:one: **pretrain AE:**

```shell
python main.py --pretrain --model pretrain_ae_for_sdcn --dataset acm --desc pretrain_ae_for_SDCN_on_acm
# or simplified command:
python main.py -P -M pretrain_ae_for_sdcn -D acm -DS pretrain_ae_for_SDCN_on_acm
```

:two: **train SDCN:**

```shell
python main.py --model SDCN --dataset acm --norm --desc Train_SDCN_1_iteration_on_the_ACM_dataset
# or simplified command:
python main.py -M SDCN -D acm -N  -DS Train_SDCN_1_iteration_on_the_ACM_dataset
```

✈️ `Step 4`: If you run the code successfully, don't forget give me a star! :wink:

### 🔓 Support Models Currently

| No.  |    Model    |                            Paper                             |                           Analysis                           |                       Source Code                       |
| :--: | :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------------------------: |
|  1   |  **DAEGC**  | [《Attributed Graph Clustering:  <br> A Deep Attentional Embedding Approach》](https://arxiv.org/pdf/1906.06532.pdf) | [论文阅读02](https://www.marigold.website/readArticle?workId=102&author=Marigold&authorId=1000001) |      [link](https://github.com/Tiger101010/DAEGC)       |
|  2   |  **SDCN**   | [《Structural Deep Clustering Network》](https://arxiv.org/pdf/2002.01633.pdf) | [论文阅读03](https://www.marigold.website/readArticle?workId=103&author=Marigold&authorId=1000001) |         [link](https://github.com/bdy9527/SDCN)         |
|  3   |  **AGCN**   | [《Attention-driven Graph Clustering Network》](https://arxiv.org/pdf/2108.05499.pdf) | [论文阅读04](https://www.marigold.website/readArticle?workId=105&author=Marigold&authorId=1000001) | [link](https://github.com/ZhihaoPENG-CityU/MM21---AGCN) |
|  4   | **EFR-DGC** | [《Deep Graph clustering with enhanced <br> feature representations for community detection》](https://link.springer.com/article/10.1007/s10489-022-03381-y) | [论文阅读12](https://www.marigold.website/readArticle?workId=140&author=Marigold&authorId=1000001) |        [link](https://github.com/grcai/DGC-EFR)         |
|  5   |  **GCAE**   |          :exclamation: ​In fact, it's GAE with GCN.           |                              -                               |                            -                            |

> In the future, I plan to update the other models. If you find my framework useful, feel free to contribute to its improvement by submitting your own code.

##  🍊 Advanced

### :exclamation: ​Arguments

#### 🥤 Help

```shell
> python main.py --help
usage: main.py [-h] [-P] [-TS] [-H] [-N] [-SLF] [-SF] [-DS DESC]
               [-M MODEL_NAME] [-D DATASET_NAME] [-R ROOT] [-K K] [-T T]
               [-LS LOOPS] [-F {tensor,npy}] [-L {tensor,npy}]
               [-A {tensor,npy}]

Scalable Unified Framework of Deep Graph Clustering

optional arguments:
  -h, --help            show this help message and exit
  -P, --pretrain        Whether to pretrain. Using '-P' to pretrain.
  -TS, --tsne           Whether to draw the clustering tsne image. Using '-TS'
                        to draw clustering TSNE.
  -H, --heatmap         Whether to draw the embedding heatmap. Using '-H' to
                        draw embedding heatmap.
  -N, --norm            Whether to normalize the adj, default is False. Using
                        '-N' to load adj with normalization.
  -SLF, --self_loop_false
                        Whether the adj has self-loop, default is True. Using
                        '-SLF' to load adj without self-loop.
  -SF, --symmetric_false
                        Whether the normalization type is symmetric. Using
                        '-SF' to load asymmetric adj.
  -DS DESC, --desc DESC  The description of this experiment.
  -M MODEL_NAME, --model MODEL_NAME
                        The model you want to run.
  -D DATASET_NAME, --dataset DATASET_NAME
                        The dataset you want to use.
  -R ROOT, --root ROOT  Input root path to switch relative path to absolute.
  -K K, --k K           The k of KNN.
  -T T, --t T           The order in GAT. 'None' denotes don't calculate the
                        matrix M.
  -LS LOOPS, --loops LOOPS
                        The Number of training rounds.
  -F {tensor,npy}, --feature {tensor,npy}
                        The datatype of feature. 'tenor' and 'npy' are
                        available.
  -L {tensor,npy}, --label {tensor,npy}
                        The datatype of label. 'tenor' and 'npy' are
                        available.
  -A {tensor,npy}, --adj {tensor,npy}
                        The datatype of adj. 'tenor' and 'npy' are available.
```

#### 🍹 Details

> Here are the details of argparse arguments you can change:

| tag  |                     arguments                      | short |                         description                          |  type/action  |  default  |
| ---- | :------------------------------------------------: | :---: | :----------------------------------------------------------: | :-----------: | :-------: |
| 🟥    |     <span style="color: red">--pretrain</span>     |  -P   |            Whether this training is pretraining.             | "store_true"  |   False   |
| 🟩    |      <span style="color: green">--tsne</span>      |  -TS  | If you want to draw the clustering result with scatter, <br> you can use it. | "store_true"  |   False   |
| 🟩    |    <span style="color: green">--heatmap</span>     |  -H   | If you want to draw the heatmap of the embedding <br> representation learned by model, you can use it. | "store_true"  |   False   |
| 🟥    |       <span style="color: red">--norm</span>       |  -N   | Whether to normalize the adj, default is False.<br> Using '-N' to load adj with normalization. | "store_true"  |   False   |
| 🟦    | <span style="color: blue">--self_loop_false</span> | -SLF  | Whether the adj has self-loop, default is True.<br> Using '-SLF' to load adj without self-loop. | "store_false" |   True    |
| 🟦    | <span style="color: blue">--symmetric_false</span> |  -SF  | Whether the normalization type is symmetric.<br> Using '-SF' to load asymmetric adj. | "store_false" |   True    |
| 🟥    |      <span style="color: red">--model</span>       |  -M   | The model you want to train.  <br>  **Should** correspond to the model in the model directory. |      str      |  "SDCN"   |
| 🟥    |     <span style="color: red">--dataset</span>      |  -D   | The dataset you want to train. <br> **Should** correspond to the dataset name in the dataset directory. |      str      |   "acm"   |
| 🟦    |        <span style="color: blue">--k</span>        |  -K   | For graph dataset, it is set to None. <br> If the dataset is not graph type, <br> you should set k to construct '**KNN**' graph of dataset. |      int      |   None    |
| 🟦    |        <span style="color: blue">--t</span>        |  -T   | If the model need to get the matrix M, such as DAEGC, <br> you should set t according to the paper. <br> None denotes the model needn't M. |      int      |   None    |
| 🟥    |      <span style="color: red">--loops</span>       |  -LS  | The training times. If you want to train the model <br> for 10 times, you can set it to 10. |      int      |     1     |
| 🟥    |       <span style="color: red">--root</span>       |  -R   | If you need to change the relative path to the <br> absolute path,  you can set it to root path. |      str      |   None    |
| 🟪    |     <span style="color: purple">--desc</span>      |  -DS  |             The description of this experiment.              |      str      | "default" |
| 🟦    |     <span style="color: blue">--feature</span>     |  -F   | The datatype of feature.<br> 'tenor' and 'npy' are available. |      str      | "tensor"  |
| 🟦    |      <span style="color: blue">--label</span>      |  -L   | The datatype of label.<br> 'tenor' and 'npy' are available.  |      str      |   "npy"   |
| 🟦    |       <span style="color: blue">--adj</span>       |  -A   |  The datatype of adj.<br> 'tenor' and 'npy' are available.   |      str      | "tensor"  |

> 💡 **Tips:**
>
> - The arguments marked with 🟥 are usually need to be specified.
> - The arguments marked with 🟩 are the drawing functions.
> - The arguments marked with 🟦 are related to the data loading.
> - The argument marked with 🟪 is strongly recommended to you to record the experimental key points.
> - **Note** that "--norm" is used in the graph convolutional network to obtain a symmetric normalized adjacency matrix, but it is not required for the graph attention network. If both are used at the same time, it is recommended to obtain the adjacency matrix without symmetric normalization first, and then manually symmetric normalize it.

### 🧩 Scalability

Strong scalability is a prominent feature of this framework. If you want to run your own code in this framework, you can follow the steps:

#### 🐯 Model Extension

🚄 `Step 1`: Write a model file `model.py` using Pytorch and a training function file `train.py` and then put them into a directory named after the uppercase of model name. Then put it into the model directory. We provide the template file in the template directory.

🚄 `Step 2`: If your model need to be pretrained, you need to write a pretraining file `train.py `and put it into a directory named after pretrain\_{module(lowercase)} \_for\_{model (lowercase)}, then put it into the model directory. We provide the template file in the template directory.

🚄 `Step 3`: Modify the pretrain_type_dict in line 38 in `path_manager.py`. The format is "model name(uppercase)": [items]. If your model needn't be pretrained, let the list null. Otherwise, you should list all modules you need to pretrain. For example, if you want to pretrain AE module, you should add "pretrain_ae" to the list. Meanwhile, please check whether the pretrain type  exists in if-else sentence, if not, please add it manually.

🚄 `Step 4`: Run your code!

#### 🐴 Dataset Extension

🚌 `Step 1`: Make sure that your dataset are well processed and the file suffix is 'npy' which denotes the file store the numpy array. If your dataset is graph data, you need to include {dataset name}\_feat.npy、{dataset name}\_label.npy、{dataset name}\_adj.npy. If your dataset is non-graph data, there are two ways to handle. One is directly using {dataset name}\_feat.npy、{dataset name}\_label.npy, and set the type of constructing graph in line 167 in `load_data.py`. If the construct type not exists, please add it to the function `construct_graph` in `data\_processor.py`. Another is to construct graph data manually, and use {dataset name}\_feat.npy、{dataset name}\_label.npy、{dataset name}\_adj.npy, but you need remember what value the k used because the dataset is considered as graph dataset.

 🚌 `Step 2`: Putting the file above to a directory named after the lowercase of dataset name. Then put them into the dataset directory.

🚌 `Step 3`: Adding the information about the dataset in the `dataset_info.py`.

🚌 `Step 4`: Use your dataset!

## 🍎 Ending

> Graph deep clustering is currently in a stage of rapid development, and more graph clustering methods will be proposed in the future. Therefore, providing a unified code framework can save researchers' coding and experiment time, and put more energy on the theoretical innovation. It is believed that graph clustering will reach a higher level in the future.
>
> If this warehouse is helpful to you, please remember to Star~😘.

