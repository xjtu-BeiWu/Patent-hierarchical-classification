# Patent-hierarchical-classification
# 数据处理

## 输入
`abstract.txt`  专利完整的摘要文本，格式：一个专利一行。

## 处理步骤

### 根据停用词表以及需要的摘要长度处理摘要文本
`generate_abstract(abstract_source, output, sentence_length)`

`abstract_source` 摘要文本路径

`output` 输出路径

`sentence_length` 需要的摘要长度

> 停用词
```
stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which'
            ]
// 标点等特殊字符 . , ; \n \\n /
```

### 根据语料和词频生成字典
`create_dic(input_path, output_path, freq)`

`input_path` 语料路径

`output_path` 输出路径

`freq` 词频，小于freq的词会被忽略

> 第一行为`,word`，最后一行为`{len(dict)},unk`

### 根据字典和处理后的语料生成语料向量
`create_sentence_matrix(abstract_path, output, dict_path, sentence_length)`

`abstract_path` 处理后的语料路径

`output` 输出路径

`dict_path` 字典路径

`sentence_length` 语料长度

> 如果长度不足`sentence_length`，会补上`unk`对应的序号

### 根据标签生成`section`对应的`one hot`向量以及对应的字典
`create_section_vector(label_path, section_vector_path, section_dict_path)`

`label_path` 标签路径

`section_vector_path` 输出`section`向量路径

`section_dict_path` 输出`section`字典路径

### 根据标签生成`subsection`对应的`one hot`向量以及对应的字典
`create_subsection_vector(label_path, subsection_vector_path, subsection_dict_path)`

`label_path` 标签路径

`subsection_vector_path` 输出`subsection`向量路径

`subsection_dict_path` 输出`subsection`字典路径

### 根据标签生成`class`对应的`one hot`向量以及对应的字典
`create_class(label_path, class_path, class_dict_path)`

`label_path` 标签路径

`class_path` 输出`class`向量路径

`class_dict_path` 输出`class`字典路径

### 合并以上处理的文件（语料 + 引用 + section + subsection）
`merge_file(abstract_vector_path, citation_vector_path, section_vector_path, subsection_vector_path, output)`

`abstract_vector_path` 语料向量

`citation_vector_path` 引用向量

`section_vector_path` `section`向量

`subsection_vector_path` `subsection`向量

`output` 输出目录

### 按8：1：1分割数据集
`split_dataset(input_path, train_data, validate_data, test_data)`

`input_path` 输入大矩阵目录

`train_data` 输出训练数据

`validate_data` 输出验证数据

`test_data` 输出测试数据
