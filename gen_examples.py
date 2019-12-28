import random as r


class LanguageGen(object):
    def __init__(self, pos_data_file_path, neg_data_file_path, sub_seq_size_lim=100):
        self._pos_data_file_path = pos_data_file_path
        self._neg_data_file_path = neg_data_file_path
        self.sub_seq_size_lim = sub_seq_size_lim

    def _gen_example(self, is_pos):
        seq = ""
        cur_sub_seq_size = r.randint(1, self.sub_seq_size_lim)
        for i in range(cur_sub_seq_size):
            seq += str(r.randint(1, 9))

        if is_pos:
            seq_parts = ['a', 'b', 'c', 'd']
        else:
            seq_parts = ['a', 'c', 'b', 'd']

        for part in seq_parts:
            cur_sub_seq_size = r.randint(1, self.sub_seq_size_lim)
            for i in range(cur_sub_seq_size):
                seq += part
            cur_sub_seq_size = r.randint(1, self.sub_seq_size_lim)
            for i in range(cur_sub_seq_size):
                seq += str(r.randint(1, 9))

        return seq

    def _gen_pos_example(self):
        return self._gen_example(True)

    def _gen_neg_example(self):
        return self._gen_example(False)

    def genExamples(self, size):
        self.db = []
        for i in range(size):
            self.db.append((self._gen_pos_example(), 0))
            self.db.append((self._gen_neg_example(), 1))
        return self.db
    

    def gen_data_files(self, size):
        with open(self._pos_data_file_path, 'w') as pos_file:
            pos_exmpl_list = []
            for i in range(size):
                exmpl = self._gen_pos_example()
                while exmpl in pos_exmpl_list:
                    exmpl = self._gen_pos_example()
                pos_exmpl_list.append(exmpl)
                if i == size - 1:
                    pos_file.write(exmpl)
                else:
                    pos_file.write(exmpl + "\n")

        with open(self._neg_data_file_path, 'w') as neg_file:
            neg_exmpl_list = []
            for i in range(size):
                exmpl = self._gen_neg_example()
                while exmpl in neg_exmpl_list:
                    exmpl = self._gen_neg_example()
                neg_exmpl_list.append(exmpl)
                if i == size-1:
                    neg_file.write(exmpl)
                else:
                    neg_file.write(exmpl + "\n")



if __name__ == "__main__":
    language_gen_t = LanguageGen(pos_data_file_path="pos_examples_train", neg_data_file_path="neg_examples_train")

    language_gen_t.gen_data_files(size=1000)

    language_gen_d = LanguageGen(pos_data_file_path="pos_examples_dev", neg_data_file_path="neg_examples_dev")

    language_gen_d.gen_data_files(size=1000)
