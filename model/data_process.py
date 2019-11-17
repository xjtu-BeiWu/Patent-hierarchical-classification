# -*- coding: utf-8 -*-
import os
import re
import json
import random

# generate word_dic from sentences
def create_dic(input_path, output_path, freq):
    f = open(input_path, 'r', encoding='utf-8')
    tmp = {}
    for line in f:
        for word in line.strip().split():
            if tmp.get(word) is None:
                tmp[word] = 1
            else:
                tmp[word] += 1
    f.close()

    wf = open(output_path, 'w', encoding='utf-8')
    wf.write(',word\n')
    tmp1 = []
    for key in tmp:
        if tmp[key] > freq:
            tmp1.append(key)
    for index, value in enumerate(tmp1):
        wf.write(str(index) + ',' + value + '\n')
    wf.write(str(len(tmp1)) + ',' + 'unk\n')
    wf.close()


def create_google_index(input_path, output_path):
    f = open(input_path, 'r', encoding='utf-8')
    lines = f.readlines()
    wf = open(output_path, 'w', encoding='utf-8')
    for i in range(1, len(lines)):
        if i % 10000 == 0:
            print(str(i / 30000) + '%')
        wf.write(lines[i].strip().split()[0] + ' ' + str(i) + '\n')
    wf.close()
    f.close()


def create_publication2application(application_all, subject_path, output_path):
    f = open(application_all, 'r', encoding='utf-8')
    subject = open(subject_path, 'r', encoding='utf-8')
    output_file = open(output_path, 'w', encoding='utf-8')
    publication2application = {}
    for line in f:
        publication2application[line.strip().split()[0]] = line.strip().split()[1]
    for line in subject:
        output_file.write(line.strip() + ' ' + publication2application[line.strip()] + '\n')
    f.close()
    subject.close()
    output_file.close()


def create_publication2citation(application, application2citation, subject_path, output_path):
    application_file = open(application, 'r', encoding='utf-8')
    application2citation_file = open(application2citation, 'r', encoding='utf-8')
    output_file = open(output_path, 'w', encoding='utf-8')
    subject_file = open(subject_path, 'r', encoding='utf-8')
    publication2application = {}
    application2citation_obj = {}
    zeros = ''
    for i in range(0, 128):
        zeros += ' 0'
    for line in application_file:
        publication2application[line.split()[0]] = line.split()[1]
    for line in application2citation_file.readlines()[1:]:
        application2citation_obj[line.strip().split()[0]] = line.replace(line.strip().split()[0], '')
    for line in subject_file:
        # print(line)
        if application2citation_obj.get(publication2application[line.strip()]) is None:
            output_file.write(line.strip() + zeros + '\n')
        else:
            output_file.write(
                line.strip() + ' ' + application2citation_obj[publication2application[line.strip()]].strip() + '\n')
    application_file.close()
    application2citation_file.close()
    output_file.close()
    subject_file.close()


def generate_abstract(abstract_source, output, sentence_length):
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
    i = 1
    abstracts = open(abstract_source, 'r', encoding='utf-8')
    output_file = open(output, 'w', encoding='utf-8')
    for line in abstracts:
        o_line = re.sub(u'[(][^)]+[)]', ' ', line)
        o_line = o_line.replace(',', ' ').replace('.', ' ').replace(';', ' ').replace('\n', ' ') \
            .replace('\\n', ' ').replace('/', ' ').lower()
        o_line = re.sub(u'\s+', ' ', o_line)
        tmp = []
        for word in o_line.split(' '):
            if word not in stoplist:
                tmp.append(word)
        output_file.write(' '.join(tmp[0:sentence_length]) + '\n')
        if i % 100 == 0:
            print(i)
        i += 1
    abstracts.close()
    output_file.close()

def extract_abstract(abstract_source, output):
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
    i = 1
    abstracts = open(abstract_source, 'r', encoding='utf-8')
    output_file = open(output, 'w', encoding='utf-8')
    for line in abstracts:
        o_line = re.sub(u'[(][^)]+[)]', ' ', line)
        o_line = o_line.replace(',', ' ').replace('.', ' ').replace(';', ' ').replace('\n', ' ') \
            .replace('\\n', ' ').replace('/', ' ').lower()
        o_line = re.sub(u'\s+', ' ', o_line)
        tmp = []
        for word in o_line.split(' '):
            if word not in stoplist:
                tmp.append(word)
        output_file.write(' '.join(tmp) + '\n')
        if i % 100 == 0:
            print(i)
        i += 1
    abstracts.close()
    output_file.close()


# too slow
def filter_dict_by_google(dict_before, dict_after, google_dict):
    dict_b = open(dict_before, 'r', encoding='utf-8')
    dict_a = open(dict_after, 'w', encoding='utf-8')
    w2c_file = open(google_dict, 'r', encoding='utf-8')
    w2c_dic = {}
    tmp = []
    index = 0
    for line in w2c_file:
        w2c_dic[line.split(' ')[0]] = 0
    print('google dict read end')
    for line in dict_b:
        print(line.split(' ')[1])
        if w2c_dic.get(line.split(' ')[1]) is not None:

            # tmp.append(line.split(' ')[1])
            dict_a.write(str(index) + ' ' + line.split(' ')[1] + '\n')
            index += 1
            if index % 100 == 0:
                print(index)
    # print('dict_b read end')
    # for i, value in tmp:
    #     dict_a.write(str(i) + ' ' + value + '\n')
    dict_a.close()
    dict_b.close()
    w2c_file.close()


def generate_dict_embedding(pretrained_path, dict_path, output_path):
    pretrained = open(pretrained_path, 'r', encoding='utf-8')
    dict = open(dict_path, 'r', encoding='utf-8')
    dict_embedding = open(output_path, 'w', encoding='utf-8')
    zeros = ''
    for i in range(0, 300):
        zeros += ' 0'
    zeros = zeros.strip()
    w2c_dict = {}
    sum = 0
    for line in pretrained.readlines()[1:]:
        w2c_dict[line.strip().split(' ')[0]] = line.strip().replace(line.strip().split(' ')[0], '')
    print('w2c builded')
    for line in dict:
        if w2c_dict.get(line.strip().split(' ')[1]) is None:
            dict_embedding.write(w2c_dict['unk'].strip() + '\n')
            sum += 1
        else:
            dict_embedding.write(w2c_dict[line.strip().split(' ')[1]].strip() + '\n')
    dict_embedding.close()
    dict.close()
    pretrained.close()
    print(sum)


def create_sentence_matrix(abstract_path, output, dict_path, sentence_length = 100):
    dict_file = open(dict_path, 'r', encoding='utf-8')
    abstract_file = open(abstract_path, 'r', encoding='utf-8')
    o_file = open(output, 'w', encoding='utf-8')
    _dict = {}
    for line in dict_file.readlines()[1:]:
        _dict[line.strip().split(',')[1]] = line.strip().split(',')[0]
    for line in abstract_file:
        v = ''
        for word in line.strip().split(' '):
            if _dict.get(word) is None:
                v += _dict['unk'] + ' '
            else:
                v += _dict[word] + ' '
        for i in range(0, sentence_length - len(line.strip().split(' '))):
            v += _dict['unk'] + ' '
        o_file.write(v.strip() + '\n')
    dict_file.close()
    abstract_file.close()
    o_file.close()


def create_section_vector(label_path, section_vector_path, section_dict_path):
    label = open(label_path, 'r', encoding='utf-8')
    v_file = open(section_vector_path, 'w', encoding='utf-8')
    d_file = open(section_dict_path, 'w', encoding='utf-8')
    section_dict = {}
    tmp = ''
    for line in label:
        for label_item in line.replace('[', '').replace(']', '').strip().split(','):
            if section_dict.get(label_item.strip()[1]) is None:
                section_dict[label_item.strip()[1]] = len(section_dict)
                tmp += str(section_dict[label_item.strip()[1]]) + ' '
            else:
                tmp += str(section_dict[label_item.strip()[1]]) + ' '
        tmp += '\n'
    print(tmp)
    for section in section_dict:
        d_file.write(section + ',' + str(section_dict[section]) + '\n')
    for line in tmp.strip().split('\n'):
        arr = line.strip().split(' ')
        temp = ''
        for index in range(0, len(section_dict)):
            if str(index) in arr:
                temp += '1 '
            else:
                temp += '0 '
        v_file.write(temp.strip() + '\n')
    label.close()
    v_file.close()
    d_file.close()


def create_subsection_vector(label_path, subsection_vector_path, subsection_dict_path):
    label = open(label_path, 'r', encoding='utf-8')
    v_file = open(subsection_vector_path, 'w', encoding='utf-8')
    d_file = open(subsection_dict_path, 'w', encoding='utf-8')
    subsection_dict = {}
    tmp = ''
    for line in label:
        for label_item in line.replace('[', '').replace(']', '').strip().split(','):
            if subsection_dict.get(label_item.strip()[1:4]) is None:
                subsection_dict[label_item.strip()[1:4]] = len(subsection_dict)
                tmp += str(subsection_dict[label_item.strip()[1:4]]) + ' '
            else:
                tmp += str(subsection_dict[label_item.strip()[1:4]]) + ' '
        tmp += '\n'
    print(tmp)
    for subsection in subsection_dict:
        d_file.write(subsection + ',' + str(subsection_dict[subsection]) + '\n')
    for line in tmp.strip().split('\n'):
        arr = line.strip().split(' ')
        temp = ''
        for index in range(0, len(subsection_dict)):
            if str(index) in arr:
                temp += '1 '
            else:
                temp += '0 '
        v_file.write(temp.strip() + '\n')
    label.close()
    v_file.close()
    d_file.close()


def create_class(label_path, class_path, class_dict_path):
    label = open(label_path, 'r', encoding='utf-8')
    c_file = open(class_path, 'w', encoding='utf-8')
    d_file = open(class_dict_path, 'w', encoding='utf-8')
    class_dict = {}
    tmp = ''
    for line in label:
        for label_item in line.replace('[', '').replace(']', '').strip().split(','):
            if class_dict.get(label_item.strip()[1:5]) is None:
                class_dict[label_item.strip()[1:5]] = len(class_dict)
                tmp += str(class_dict[label_item.strip()[1:5]]) + ' '
            else:
                tmp += str(class_dict[label_item.strip()[1:5]]) + ' '
        tmp += '\n'
    print(tmp)
    for class_ in class_dict:
        d_file.write(class_ + ',' + str(class_dict[class_]) + '\n')
    for line in tmp.strip().split('\n'):
        arr = line.strip().split(' ')
        temp = ''
        for index in range(0, len(class_dict)):
            if str(index) in arr:
                temp += '1 '
            else:
                temp += '0 '
        c_file.write(temp.strip() + '\n')
    label.close()
    c_file.close()
    d_file.close()


def merge_file(abstract_vector_path, citation_vector_path, section_vector_path, subsection_vector_path, output):
    abstract_vector = open(abstract_vector_path, 'r', encoding='utf-8').readlines()
    citation_vector = open(citation_vector_path, 'r', encoding='utf-8').readlines()
    section_vector = open(section_vector_path, 'r', encoding='utf-8').readlines()
    subsection_vector = open(subsection_vector_path, 'r', encoding='utf-8').readlines()
    o_file = open(output, 'w', encoding='utf-8')
    for index in range(0, len(section_vector)):
        o_file.write(abstract_vector[index].strip() + ' ' + ' '.join(citation_vector[index].strip().split(' ')[1:]).strip() + ' '
                     + section_vector[index].strip() + ' ' + subsection_vector[index].strip() + '\n')
    o_file.close()


def generate_baseline_data(abstract_path, label_dict_path, output_path):
    abstracts = open(abstract_path, 'r', encoding='utf-8').readlines()
    labels = open(label_dict_path, 'r', encoding='utf-8').readlines()
    o_file = open(output_path, 'w', encoding='utf-8')
    for i in range(0, len(abstracts)):
        tmp = {}
        tmp['testid'] = str(i)
        tmp['features_content'] = abstracts[i].strip().split(' ')
        tmp['labels_index'] = []
        for index, key in enumerate(labels[i].split(' ')):
            if key == '1':
                tmp['labels_index'].append(index)
        tmp['labels_num'] = len(tmp['labels_index'])
        o_file.write(json.dumps(tmp) + '\n')
    o_file.close()


def split_dataset(input_path, train_data, validate_data, test_data):
    data = open(input_path, 'r', encoding='utf-8').readlines()
    train = open(train_data, 'w', encoding='utf-8')
    validate = open(validate_data, 'w', encoding='utf-8')
    test = open(test_data, 'w', encoding='utf-8')
    random.shuffle(data)
    for line in data[0:180718]:
        validate.write(line.strip() + '\n')
    validate.close()
    for line in data[180718:361436]:
        test.write(line.strip() + '\n')
    test.close()
    for line in data[361436:]:
        train.write(line.strip() + '\n')
    train.close()


if __name__ == '__main__':
    data_path = '/Volumes/Storage1T/firPro/'
    # create_dic(os.path.join(data_path, 'abstract-100.txt'), os.path.join(data_path, 'dict_100_5.txt'), 5)
    # create_google_index(os.path.join(data_path, 'GoogleNews-vectors-negative300.txt'),
    #                     os.path.join(data_path, 'google-dic.txt'))
    # create_publication2application(os.path.join(data_path, 'application_all.txt'),
    #                                os.path.join(data_path, 'subject.txt'),
    #                                os.path.join(data_path, 'application.txt'))

    # create_publication2citation(os.path.join(data_path, 'application.txt'),
    #                             os.path.join(data_path, 'application_citations_embeddings_order2_epoch50.txt'),
    #                             os.path.join(data_path, 'subject.txt'),
    #                             os.path.join(data_path, 'publication_citations_embeddings_order2_epoch50.txt'))
    # generate_abstract(os.path.join(data_path, 'abstract.txt'),
    #                   os.path.join(data_path, 'abstract_100_new.txt'),
    #                   100)
    # extract_abstract(os.path.join(data_path, 'abstract.txt'),
    #                   os.path.join(data_path, 'abstract_extracted.txt'))
    # 构建字典
    # create_dic(os.path.join(data_path, 'abstract_100_new.txt'), os.path.join(data_path, 'dict_100_5.csv'), 5)
    # filter_dict_by_google(os.path.join(data_path, 'dict_100_5_new_before_filter_by_google.txt'),
    #                       os.path.join(data_path, 'dict_100_5_new_after.txt'),
    #                       os.path.join(data_path, 'google-dic.txt'))
    # generate_dict_embedding(os.path.join(data_path, 'GoogleNews-vectors-negative300.txt'),
    #                         os.path.join(data_path, 'dict_100_5_new.txt'),
    #                         os.path.join(data_path, 'dict_w2c.txt'))
    # 构建句子矩阵
    # create_sentence_matrix(os.path.join(data_path, 'abstract_100_new.txt'),
    #                        os.path.join(data_path, 'abstract_100_matrix.txt'),
    #                        os.path.join(data_path, 'dict_100_5.csv'))
    # 生成section向量
    # create_section_vector(os.path.join(data_path, 'label.txt'),
    #                       os.path.join(data_path, 'section_vector.txt'),
    #                       os.path.join(data_path, 'section_dict.txt'))
    # 生成subsection向量
    # create_subsection_vector(os.path.join(data_path, 'label.txt'),
    #                       os.path.join(data_path, 'subsection_vector.txt'),
    #                       os.path.join(data_path, 'subsection_dict.txt'))
    # 生成class向量
    # create_class(os.path.join(data_path, 'label.txt'),
    #              os.path.join(data_path, 'class_vector.txt'),
    #              os.path.join(data_path, 'class_dict.txt'))
    # 合并向量
    # merge_file(os.path.join(data_path, 'abstract_100_matrix.txt'),
    #            os.path.join(data_path, 'publication_citations_embeddings_order2_epoch50.txt'),
    #            os.path.join(data_path, 'section_vector.txt'),
    #            os.path.join(data_path, 'subsection_vector.txt'),
    #            os.path.join(data_path, 'all_vector.txt'))
    # generate_baseline_data(os.path.join(data_path, 'abstract_100_new.txt'),
    #                        os.path.join(data_path, 'class_vector.txt'),
    #                        os.path.join(data_path, 'Train.txt'))
    split_dataset(os.path.join(data_path, 'Train.txt'),
                  os.path.join(data_path, 'Train.json'),
                  os.path.join(data_path, 'Validation.json'),
                  os.path.join(data_path, 'Test.json'))