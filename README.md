# A-Unified-Framework-for-Deep-Attribute-Graph-Clustering

See the Chinese version in [Marigold博客]()

Recently, attribute graph clustering has developed rapidly, at the same time various deep attribute graph clustering methods have sprung up. Although most of the methods are open source, it is a pity that these codes do not have a unified framework, which makes researchers have to spend a lot of time modifying the code to achieve the purpose of reproduction. Fortunately, Liu et al. \[Homepage: [yueliu1999](https://github.com/yueliu1999)\] organized the deep graph clustering method into a code warehouse—— [ Awesome-Deep-Graph-Clustering(ADGC)](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering). For example, they provided more than 20 datasets and unified the format. Moreover, they list the most related paper about deep graph clustering  and give the link of source code. It is worth mentioning that they organize the code of deep graph clustering into rand-augmentation-model-clustering-visualization-utils structure, which greatly facilitates beginners and researchers. Here, on behalf of myself, I would like to express my sincere thanks and high respect to Liu et al. ❤️

**Acknowledgements:**

Thanks for the open source of these authors (not listed in order):

\[ [yueliu1999](https://github.com/yueliu1999) | [bdy9527](https://github.com/bdy9527)| [Tiger101010](https://github.com/Tiger101010)\]

## Introduction

On the basis of ADGC, I refactored the code to make the deep clustering code achieve a higher level of unification. Specifically, I redesigned the architecture of the code, so that you can run the open source code easily. I defined some tool classes and functions to simplify the code and make the settings' configuration clear.  

- `main.py`: The **entrance** file of my framework.
- `requirements.txt`: The third-party library environments that need to be installed first.
- `dataset`: The directory including the dataset you need, whose subdirectories are named after dataset names. The subdirectory includes the features file, the labels file and the adjacency matrix file, named with **{dataset name}\_feat.npy**, **{dataset name}\_label.npy** and **{dataset name}\_adj.npy**, such as **acm\_feat.npy**, **acm\_label.npy** and **acm\_adj.npy**. **Besides**, the dataset directory also includes a python file named dataset\_info.py which stores the information related to datasets.
- `module`: The directory including the most used basic modules of model, such as the Auto-encoder (**AE.py**), the Graph Convolutional Layer (**GCN.py**), the Graph Attention Layer (**GAT.py**), et al.
- `model`: The directory including the model you want to run. The directory format is a subdirectory named after the uppercase letters of the model name, which contains two files, one is the model file **model.py** for storing model classes, and the other is the training file **train.py** for model training. Our framework will dynamically import the training file of the model according to the input model name. **Besides**, it can also store the pre-training directory named the lowercase letters of pretrain\_{module name}\_for\_{model name}, which stores the **train.py** file. For example, if you want to pretrain the AE module in SDCN, you can named the directory with **pretrain\_ae\_for\_sdcn**. 
- `utils`: The directory including some **tool** classes and functions.
  - `load_data.py`: It includes the functions of  **loading dataset** for training.
  - `data_processor.py`: It includes the functions of transferring data storing types and others, such as **numpy to torch**, **symmetric normalization** et al.
  - `calculator.py`: It includes the function of calculating **mean** and standard difference.
  - `evalution.py`: It includes the function of calculating the related **metrics** of clustering, such as ACC, NMI, ARI and F1\_score.
  - `formatter.py`: It includes the function of **formatting** the output of **variables** according to your input variables.
  - `logger.py`: It includes **a log class**, through which you can record the information you want to output to the log file.
  - `parameter_counter.py`: It includes the function of counting the model's parameters.
  - `path_manager.py`: It includes the function of  transforming **the relative path** to **the absolute path** if needed. Of course, if you don't need, it also should be called because it also **stores the path** needed by the training, such as the storing path of logs, pretrain parameters files, clustering visualization images, et al.
  - `plot.py`: It includes the function of drawing clustering visualization via **TSNE** and save the image. The features heatmap will also be developed soon later.
  - `time_manager.py`: It includes **a time class** to record time consuming and a function to format datetime.
- `logs`: The directory is used to **store the output logs files**. Its subdirectories are named after the model names and the logs files are named after the start time.
- `pretrain`:  The directory is used to **store the pre-training parameters files**. Its subdirectories are named after the format of pretrain\_{model name}. Parameters files are categorized by model and dataset name.
- `img`: The directory is used to **store the output images**, whose subdirectories are named after **clustering** and **heatmap**.

## Quick Start

After git clone the code, you can follow the steps below to run:

`Step 1`: Check the environment or run the requirements.txt to install the libraries directly.

```bash
pip install -r requirements.txt
```

`Step 2`: Prepare the datasets. If you don't have the datasets, you can download them from Liu's warehouse \[[yueliu1999](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering) | [Google Drive](https://drive.google.com/drive/folders/1thSxtAexbvOyjx-bJre8D4OyFKsBe1bK) | [Nutstore](https://www.jianguoyun.com/p/DfzK1pwQwdaSChjI2aME)\]. Then unzip them to the dataset directory.

`Step 3`: Run the file in the directory where main.py is located in command line. If it is in the integrated compilation environment, you can directly run the main.py file. 

Take the training of DAEGC as example:

pretrain GAT:

```shell
python main.py --is_pretrain True --model_name pretrain_gat_for_daegc --dataset_name acm
```

train DAEGC:

```shell
python main.py --model_name DAEGC --dataset_name cora
```

Take the training of SDCN as example:

pretrain AE:

```shell
python main.py --is_pretrain True --model_name pretrain_ae_for_sdcn --dataset_name acm
```

train SDCN:

```shell
python main.py --model_name SDCN --dataset_name acm
```

> Here are the argparse arguments you can change:

|        arguments         |                         description                          | type | default |
| :----------------------: | :----------------------------------------------------------: | :--: | :-----: |
|      --is\_pretrain      |            Whether this training is pre-training.            | bool |  False  |
|      --model\_name       | The model you want to train.  <br>  **Should** correspond to the model in the model directory. | str  |  DAEGC  |
|     --dataset\_name      | The dataset you want to train. <br> **Should** correspond to the dataset name in the dataset directory. | str  |   acm   |
|           --k            | For graph dataset, it is set to None. <br> If the dataset is not graph type, <br> you should set k to construct '**KNN**' graph of dataset. | int  |  None   |
|           --t            | If the model need to get the matrix M, such as DAEGC, <br> you should set t according to the paper. | int  |    2    |
|    --update_interval     |                     The update interval.                     | int  |    1    |
|         --loops          | The training times. If you want to train the model <br> for 10 times, you can set it to 10. | int  |    1    |
|  --is\_change_root_path  | If you need to change the relative path to the <br> absolute path,  you can set it to True. <br> **Note that** it just a flag, the root path <br> should be changed manually in the main.py. Then the <br> programmer will auto change all the path. | bool |  False  |
| --plot\_clustering\_tsne | If you want to draw the clustering result with scatter, <br> you can set it to True. **Note that** don't use it in <br>multiple loops, or you can only get the last loop result. | bool |  False  |
| --plot_embedding_heatmap | If you want to draw the heatmap of the embedding <br> representation learned by model, you can set it to True. | bool |  False  |

`Step 4`: If you run the code successfully, don't forget give me a star! :wink:

## Support Models Currently

| No.  |   Model   |                            Paper                             |                           Analysis                           |                 Source Code                  |
| :--: | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :------------------------------------------: |
|  1   | **DAEGC** | [《Attributed Graph Clustering:  <br> A Deep Attentional Embedding Approach》](https://arxiv.org/pdf/1906.06532.pdf) | [论文阅读02](https://www.marigold.website/readArticle?workId=102&author=Marigold&authorId=1000001) | [link](https://github.com/Tiger101010/DAEGC) |
|  2   | **SDCN**  | [《Structural Deep Clustering Network》](https://arxiv.org/pdf/2002.01633.pdf) | [论文阅读03](https://www.marigold.website/readArticle?workId=103&author=Marigold&authorId=1000001) |   [link](https://github.com/bdy9527/SDCN)    |

> In the future, I plan to update the other models. If you find my framework useful, feel free to contribute to its improvement by submitting your own code.

