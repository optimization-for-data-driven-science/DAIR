import json
import os
import os.path
import re

from PIL import Image
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import config

import utils
import ipdb
import numpy as np
import random


# orig_edit_combine=1

def get_loader(train=False, val=False, test=False, prefix=''):   #TODO you need todo some changes here, this here decides what is the data thta goes into loader while train/test
    """ Returns a data loader for the desired split """
    assert train + val + test == 1, 'need to set exactly one of {train, val, test} to True'
    split = VQA(
        utils.path_for(train=train, val=val, test=test, question=True,prefix = prefix),
        utils.path_for(train=train, val=val, test=test, answer=True, prefix = prefix),
        config.preprocessed_path,     ## make changed here- in config file  #TODO you need todo some changes here, this here decides what is the data thta goes into loader while train/test
        answerable_only=train
    )
    #print('LEN SPLIT: ',len(split))
    if config.orig_edit_equal_batch:
        batch_size_given = int(config.batch_size * config.orig_amt)
    else:
        batch_size_given = config.batch_size

    loader = torch.utils.data.DataLoader(
        split,
        batch_size=batch_size_given,
        shuffle= train, #train,   #TODO vedika  you dont want to shuffle train during test!!  #train, #train,  # only shuffle the data in training #TODO vedika comment- good idea!
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_fn,
    )
    #ipdb.set_trace()    ## check what the loader is retunring here   #len(loader.dataset) = 8438 for what color is the- works correctly- so all good- so far
    #print("done data")
    return split, loader



def get_edit_train_batch(dataset, ques_id_batch, item_ids):
    # dataset = VQA(
    #     utils.path_for(train=train, val=val, test=test, question=True),
    #     utils.path_for(train=train, val=val, test=test, answer=True),
    #     config.preprocessed_path,     ## make changed here- in config file  #TODO you need todo some changes here, this here decides what is the data thta goes into loader while train/test
    #     answerable_only=train
    #
    if config.orig_edit_diff_ratio_naive:
        return dataset._get_random_edit_batch_sample_rato_experiment()
    if config.orig_edit_diff_ratio_naive_no_edit_ids_repeat:
        return dataset._get_random_edit_batch_sample_rato_experiment_no_edit_ids_repeat()
    if config.edit_loader_type == 'get_edits':
        return dataset._get_corresponding_editIQA_batch(ques_id_batch)
    if config.edit_loader_type == 'get_all_edits':
        return dataset._get_all_corresponding_editIQA_batch(ques_id_batch)
    elif config.edit_loader_type == 'get_more_edits_if_not_64':
        return dataset._get_corresponding_editIQA_batch_get_more_edits_if_not_64(ques_id_batch)
    elif config.edit_loader_type == 'get_edits_if_not_orig':
        return dataset._get_corresponding_editIQA_batch_if_not_get_orig(ques_id_batch, item_ids)




def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)


class VQA(data.Dataset):
    """ VQA dataset, open-ended """
    def __init__(self, questions_path, answers_path, image_features_path, answerable_only=False):
        super(VQA, self).__init__()
        with open(questions_path, 'r') as fd:
            questions_json = json.load(fd)
        with open(answers_path, 'r') as fd:
            answers_json = json.load(fd)
        with open(config.vocabulary_path, 'r') as fd:
            vocab_json = json.load(fd)
        self._check_integrity(questions_json, answers_json)
        
        # vocab
        self.vocab = vocab_json
        self.token_to_index = self.vocab['question']
        self.answer_to_index = self.vocab['answer']

        # q and a
        self.questions = list(prepare_questions(questions_json))
        self.answers = list(prepare_answers(answers_json))
        print('Done prep')
        self.questions = [self._encode_question(q) for q in self.questions]
        self.answers = [self._encode_answers(a) for a in self.answers]
       
        self.image_features_path = image_features_path
        self.coco_id_to_index = self._create_coco_id_to_index()

        self.coco_ids = [q['image_id'] for q in questions_json['questions']]         ### so image_id retrieved from json files
        self.ques_ids = [q['question_id'] for q in questions_json['questions']]
        
        
        self.orig_IQA_list = [idx for idx, q in enumerate(questions_json['questions']) if len(str(q['image_id']))!=25]
        self.edit_IQA_list = [idx for idx, q in enumerate(questions_json['questions']) if len(str(q['image_id'])) == 25]
        assert len(self.orig_IQA_list) + len(self.edit_IQA_list) == len(questions_json['questions'])
        #self.orig_IQA_list = self.orig_IQA_list[:500]
        
        if config.orig_edit_equal_batch:
            orig_ques_ids = self.ques_ids[0:len(self.orig_IQA_list)]
            edit_ques_ids = self.ques_ids[len(self.orig_IQA_list):]
            from collections import defaultdict
            self.orig_edit_qid = defaultdict(list)
            self.is_edit_aug = dict()
            for idx, edit_id in enumerate(edit_ques_ids):
        #        if edit_id in orig_ques_ids:
                self.orig_edit_qid[edit_id].append(idx+len(self.orig_IQA_list))
                self.is_edit_aug[edit_id] = 1
               # else:
               #     print(edit_id)
                #if orig_id ==edit_id:
                 #   self.orig_edit_qid[orig_id].append(idx+len(self.orig_IQA_list))
            #json.dump(self.orig_edit_qid,open('./self_orig_edit_qid.json','w'))
            
            for idx, orig_id in enumerate(orig_ques_ids):
                if not self.orig_edit_qid.get(orig_id, None):
                    self.orig_edit_qid[orig_id].append(idx)
                    self.is_edit_aug[orig_id] = 0
            print('Length of actual edits available: {}'.format(sum(self.is_edit_aug.values())))
            print('length of original ids: {}, of which {} ids have corresponding edit'.format(len(self.orig_IQA_list), len(self.orig_edit_qid)))
            
        self.answerable = None
        # only use questions that have at least one answer?
        self.answerable_only = answerable_only
        if self.answerable_only:
            self.answerable = self._find_answerable()
            if config.load_only_orig_ids:
                self.answerable_orig = [i for i in self.answerable if i< len(self.orig_IQA_list) ]
            
      




    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions))
        return self._max_length

    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0   #TODO vedika where does this come from <unk> token at index-0

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        #print( 'coco_ids from features file', coco_ids)
        return coco_id_to_index

    def _check_integrity(self, questions, answers):
        """ Verify that we are using the correct data """
        #ipdb.set_trace()
        qa_pairs = list(zip(questions['questions'], answers['annotations']))
        assert all(q['question_id'] == a['question_id'] for q, a in qa_pairs), 'Questions not aligned with answers'
        assert all(q['image_id'] == a['image_id'] for q, a in qa_pairs), 'Image id of question and answer don\'t match'
        #assert questions['data_type'] == answers['data_type'], 'Mismatched data types'
        #assert questions['data_subtype'] == answers['data_subtype'], 'Mismatched data subtypes'

    def _find_answerable(self):
        """ Create a list of indices into questions that will have at least one answer that is in the vocab """
        answerable = []
        for i, answers in enumerate(self.answers):
            answer_has_index = len(answers.nonzero()) > 0
            # store the indices of anything that is answerable
            if answer_has_index:
                answerable.append(i)
        #ipdb.set_trace()
        return answerable

    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.zeros(self.max_question_length).long()
        for i, token in enumerate(question):
            index = self.token_to_index.get(token, 0)
            vec[i] = index
        return vec, len(question)

    def _encode_answers(self, answers):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = torch.zeros(len(self.answer_to_index))
        for answer in answers:
            index = self.answer_to_index.get(answer)
            if index is not None:
                answer_vec[index] += 1
        return answer_vec

    def _load_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')

        # print(self.coco_id_to_index[np.string_(image_id)])
        # print('type of image_id ',  type(image_id))                                    # type of image_id  <class 'str'>
        # print('type of image_id ',  type(np.string_(image_id)))                        # type of image_id  <class 'numpy.bytes_'>
        # print('type of image_id ',  type(np.string_(image_id).astype(np.bytes_)))      # type of image_id  <class 'numpy.bytes_'>

        index = self.coco_id_to_index[np.string_(image_id)]
        dataset = self.features_file['features']
        img = dataset[index].astype('float32')

        return torch.from_numpy(img)


    def _get_corresponding_editIQA(self, item):
        q, q_length = self.questions[item]
        a = self.answers[item]
        ques_id = self.ques_ids[item]
        image_id = self.coco_ids[item]
        if isinstance(image_id, int):
            image_id = str(image_id).zfill(12)
        v = self._load_image(image_id)
        return v, q, a, item, image_id, ques_id, q_length


    def _get_random_edit_batch_sample_rato_experiment(self):
        big_batch = [[] for list_min in range(7)]
        edit_batch_size = config.batch_size - int(config.orig_amt*config.batch_size)
        chosen_64 = np.random.choice(self.edit_IQA_list , edit_batch_size, replace=False)
        for item in chosen_64:
            if item in self.answerable:
                batch = self._get_corresponding_editIQA(item)  ## batch[0,1,2] is tensor  [3,5,6] is not [4]- image_id
                final_batch = [big_batch[i].append(batch[i]) for i in [0, 1, 2, 4]]
                final_batch = [big_batch[i].append(torch.tensor(batch[i])) for i in [3, 5, 6]]
        v, q, a, item,  ques_id, q_length = [torch.stack(big_batch[i], dim=0) for i in [0,1,2,3,5,6]]
        image_id = big_batch[4]
        if len(self.orig_IQA_list) <  len(self.edit_IQA_list):
            self.edit_IQA_list = list(set(self.edit_IQA_list) - set(chosen_64))
        return v, q, a, item, image_id, ques_id, q_length



    def _get_random_edit_batch_sample_rato_experiment_no_edit_ids_repeat(self):
        big_batch = [[] for list_min in range(7)]
        edit_batch_size = config.batch_size - int(config.orig_amt*config.batch_size)
        chosen_64 = np.random.choice(self.edit_IQA_list , edit_batch_size, replace=False)
        for item in chosen_64:
            if item in self.answerable:
                batch = self._get_corresponding_editIQA(item)  ## batch[0,1,2] is tensor  [3,5,6] is not [4]- image_id
                final_batch = [big_batch[i].append(batch[i]) for i in [0, 1, 2, 4]]
                final_batch = [big_batch[i].append(torch.tensor(batch[i])) for i in [3, 5, 6]]
        v, q, a, item,  ques_id, q_length = [torch.stack(big_batch[i], dim=0) for i in [0,1,2,3,5,6]]
        image_id = big_batch[4]
        self.edit_IQA_list = list(set(self.edit_IQA_list) - set(chosen_64))
        if len(self.edit_IQA_list) < edit_batch_size:
            self.edit_IQA_list = [i for i in range(len(self.orig_IQA_list),len(self.coco_ids))]   ## so this is stored to what it was before
        return v, q, a, item, image_id, ques_id, q_length



    def _get_corresponding_editIQA_batch(self, ques_ids_batch):
        big_batch = [[] for list_min in range(7)]
        is_edit_batch = []
        for ques_id in ques_ids_batch:
            if int(ques_id) in self.orig_edit_qid.keys():
                item = random.choice(self.orig_edit_qid[int(ques_id)])
                is_edit = self.is_edit_aug[int(ques_id)]
                is_edit_batch.append(is_edit)
                if not self.answerable or item in self.answerable:
                    batch =  self._get_corresponding_editIQA(item)   ## batch[0,1,2] is tensor  [3,5,6] is not [4]- image_id
                    final_batch = [big_batch[i].append(batch[i]) for i in [0,1,2,4]]
                    final_batch = [big_batch[i].append(torch.tensor(batch[i])) for i in [3,5,6]]
        if len(big_batch[4])!=0: # making sure that the list isn't empty- so that you can stack- big_batch[4] corresponds to image_id
            v, q, a, item,  ques_id, q_length = [torch.stack(big_batch[i], dim=0) for i in [0,1,2,3,5,6]]
            image_id = big_batch[4]
            return v, q, a, item, image_id, ques_id, q_length, is_edit_batch
        else:
            return None



    def _get_all_corresponding_editIQA_batch(self, ques_ids_batch):
        big_batch = [[] for list_min in range(7)]
        for ques_id in ques_ids_batch:
            if int(ques_id) in self.orig_edit_qid.keys():
                for item in self.orig_edit_qid[int(ques_id)]:
                    if item in self.answerable:
                        batch = self._get_corresponding_editIQA(item)  ## batch[0,1,2] is tensor  [3,5,6] is not [4]- image_id
                        final_batch = [big_batch[i].append(batch[i]) for i in [0, 1, 2, 4]]
                        final_batch = [big_batch[i].append(torch.tensor(batch[i])) for i in [3, 5, 6]]
        if len(big_batch[4])!=0: # making sure that the list isn't empty- so that you can stack- big_batch[4] corresponds to image_id
            v, q, a, item,  ques_id, q_length = [torch.stack(big_batch[i], dim=0) for i in [0,1,2,3,5,6]]
            image_id = big_batch[4]
            return v, q, a, item, image_id, ques_id, q_length
        else:
            return None



    def _get_corresponding_editIQA_batch_get_more_edits_if_not_64(self, ques_ids_batch):
        big_batch = [[] for list_min in range(7)]
        while len(big_batch[4])!= config.batch_size - int(config.batch_size * config.orig_amt):  #TODO whule lop is super slow- some other way- think of it
            for ques_id in ques_ids_batch:
                if int(ques_id) in self.orig_edit_qid.keys():
                    item = random.choice(self.orig_edit_qid[int(ques_id)])
                    if item in self.answerable:
                        batch = self._get_corresponding_editIQA(
                            item)  ## batch[0,1,2] is tensor  [3,5,6] is not [4]- image_id
                        final_batch = [big_batch[i].append(batch[i]) for i in [0, 1, 2, 4]]
                        final_batch = [big_batch[i].append(torch.tensor(batch[i])) for i in [3, 5, 6]]
                print(1)
        #ipdb.set_trace()
        v, q, a, item,  ques_id, q_length = [torch.stack(big_batch[i], dim=0) for i in [0,1,2,3,5,6]]
        image_id = big_batch[4]
        return v, q, a, item, image_id, ques_id, q_length



    def _get_corresponding_editIQA_batch_if_not_get_orig(self, ques_ids_batch, item_ids):
        big_batch = [[] for list_min in range(7)]
        for ques_id_idx, ques_id in enumerate(ques_ids_batch):
            if int(ques_id) in self.orig_edit_qid.keys():
                item = random.choice(self.orig_edit_qid[int(ques_id)])
                ## batch[0,1,2] is tensor  [3,5,6] is not [4]- image_id
            else:                                       ##return the original image and that itself? that way- enforcing consostency also holds- fair too i feel
                item = item_ids[ques_id_idx]
            if item in self.answerable:
                batch = self._get_corresponding_editIQA(item)  ## batch[0,1,2] is tensor  [3,5,6] is not [4]- image_id
                final_batch = [big_batch[i].append(batch[i]) for i in [0, 1, 2, 4]]
                final_batch = [big_batch[i].append(torch.tensor(batch[i])) for i in [3, 5, 6]]
        v, q, a, item,  ques_id, q_length = [torch.stack(big_batch[i], dim=0) for i in [0,1,2,3,5,6]]
        image_id = big_batch[4]
        ### IN THIS CASE: Q_LEN- SHOULD BE EXACT REPLICA [64] == [64]: EXCEPT item, IMAGE_ID, V: ques/ans/ques_id,q_len- just replicated- migth want to check that-
        ## also sorting use a careful trick- as you would want to enforce the consistency on these
        return v, q, a, item, image_id, ques_id, q_length



    def __getitem__(self, item):
        if self.answerable_only:
            # change of indices to only address answerable questions
            item = self.answerable[item]

        q, q_length = self.questions[item]
        a = self.answers[item]
        ques_id = self.ques_ids[item]   ## added by vedika
        image_id = self.coco_ids[item]   ## added by vedika
        if isinstance(image_id, int):
            image_id = str(image_id).zfill(12)   # Dont know why it is still len(25) - had converted to 25!
        #print(image_id)
        v = self._load_image(image_id)
        # since batches are re-ordered for PackedSequence's, the original question order is lost
        # we return `item` so that the order of (v, q, a) triples can be restored if desired
        # without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.
        return v, q, a, item, image_id, ques_id, q_length    ############### appending ques_id and img_id- vedika!!


    def __len__(self):
        if self.answerable_only:    ### is the case for training
            if config.load_only_orig_ids:
                return len(self.answerable_orig)
            else:
                return len(self.answerable)   ## returnsthose ids which have answer which is in our vocab
        else:
            # if config.load_only_orig_ids:
            #     return len(self.questions[0:len(self.orig_IQA_list)])
            # else:
            #return len(self.questions)   ## so this gives the entire list - all idx
            return len(self.questions[0:len(self.orig_IQA_list)])

# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def prepare_questions(questions_json):
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    questions = [q['question'] for q in questions_json['questions']]
    for question in questions:
        question = question.lower()[:-1]
        yield question.split(' ')


def prepare_answers(answers_json):
    """ Normalize answers from a given answer json in the usual VQA format. """       #TODO normalize answers he mentioined in github README
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
    # The only normalization that is applied to both machine generated answers as well as
    # ground truth answers is replacing most punctuation with space (see [0] and [1]).
    # Since potential machine generated answers are just taken from most common answers, applying the other
    # normalizations is not needed, assuming that the human answers are already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96

    def process_punctuation(s):
        # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
        # this version should be faster since we use re instead of repeated operations on str's
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()

    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))


class CocoImages(data.Dataset):    ## USAGE: ONLY IN PREPROCESSING IMAGES!!!
    """ Dataset for MSCOCO images located in a folder on the filesystem """
    def __init__(self, path, transform=None):
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids =  sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue

            parts_name = filename.split('_')
            # ['COCO', 'val2014', '000000177529.jpg']     # ['COCO', 'val2014', '000000177529', '000000000062.jpg']
            if len(parts_name) == 4:  # 0,1,2,3
                id_and_extension = parts_name[2] + '_' + parts_name[3]   # '000000177529_000000000062'
            else:
                id_and_extension = parts_name[2]
            id = id_and_extension.split('.')[0]

            if len(id)!=25:               ## here i take care of storing every id as string of 25 letters- S25 fixed
                id.zfill(25)

            id_to_filename[id] = filename     # {'000000177529_000000000062': 'COCO_val2014_000000177529_000000000062.jpg'}

        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]   ### sorts it here -_- TODO vedika line 233 in class CocoImages-changed it myself
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)   ####self.sorted_ids[0:10]  TODO Vedika just pass 10 images here to check what is happening


class Composite(data.Dataset):
    """ Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        current = self.datasets[0]
        for d in self.datasets:
            if item < len(d):
                return d[item]
            item -= len(d)
        else:
            raise IndexError('Index too large for composite dataset')

    def __len__(self):
        return sum(map(len, self.datasets))
