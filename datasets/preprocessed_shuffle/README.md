We prepared a simple dataset on [Google Drive](https://drive.google.com/file/d/10NDQy0IZXupqXBhKfm3j7SwF08JBrE-w/view?usp=sharing). Please download and then unzip it on the project root path.

```shell
$ tar -zxvf preprocessed_shuffle.tar.gz
```

The structure of preprocessed_shuffle directory is:

```shell
./datasets/preprocessed_shuffle/task-{Client_ID}-{Task_ID}/{train, query, or gallery}/{Person_ID}/{Image_Name}.jpg
```

Please remember to revise the `datasets_dir='./datasets/preprocessed_shuffle/'` of configuration `./configs/common.yaml`.

