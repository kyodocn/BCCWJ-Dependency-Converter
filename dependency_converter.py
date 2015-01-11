# -*- coding: utf-8 -*-

import os
import sys
import re
import codecs
from optparse import OptionParser



ContentPOS = [
    u"URL",  u"代名詞", u"副詞",   u"助動詞", u"動詞",
    u"名詞", u"形容詞", u"形状詞", u"接尾辞", u"英単語"
]


class CdSentence(object):
    def __init__(self, clauses, file_name=None, sentence_index=None):
        self.clauses = clauses
        self.dep_table = self.__construct_dep_table(clauses)
        self.file_name = file_name
        self.sentence_index = sentence_index

    @classmethod
    def read_from_lines(cls, lines, file_name=None, sentence_index=None):
        clauses = []
        clause_lines = []
        is_bos = True
        begin_index = 0
        incf = 0
        for line in lines:
            if len(line) >= 2 and line[0:2] == u"* ":
                if is_bos:
                    clause_lines.append(line)
                    is_bos = False
                else:
                    clause = Clause.read_from_lines(clause_lines, begin_index)
                    clauses.append(clause)
                    clause_lines = [line]
                    begin_index += incf
                    incf = 0
            else:
                clause_lines.append(line)
                incf += 1
        clause = Clause.read_from_lines(clause_lines, begin_index)
        clauses.append(clause)
        return cls(clauses, file_name, sentence_index)

    def __construct_dep_table(self, clauses):
        dep_table = []
        for i, clause in enumerate(clauses):
            if i != clause.clause_index:
                raise RuntimeError("clause index is invalid")
            dep_table.append(clause.dep_target_index)
        return dep_table

    def has_dummy_dependent(self):
        for target in self.dep_table[:-1]:
            if target == -1:
                return True
        return False

    def fix_dummy_dependent(self):
        for i, target in enumerate(self.dep_table[:-1]):
            if target == -1:
                self.dep_table[i] = len(self.dep_table) - 1

    def has_reverse_dependent(self):
        for i, target in enumerate(self.dep_table):
            if target != -1 and i >= target:
                return True
        return False

    def is_projective(self):
        undirected_dep_table = self.__create_undirected_dep_table()
        stack = [0]
        for i, targets in enumerate(undirected_dep_table):
            pre_target = stack[-1]
            if i == pre_target:
                while len(stack) >= 1 and stack[-1] == i:
                    stack.pop()
                stack.extend(targets)
            else:
                if any(target > pre_target for target in targets):
                    return False
                stack.extend(targets)
        return True

    def __create_undirected_dep_table(self):
        c_n = len(self.clauses)
        undirected_dep_table = [ [] for i in range(c_n) ]
        for i, target in enumerate(self.dep_table):
            if target == -1:
                undirected_dep_table[i].append(c_n)
            elif i > target:
                undirected_dep_table[target].append(i)
            else:
                undirected_dep_table[i].append(target)
        for targets in undirected_dep_table:
            targets.sort(reverse=True)
        return undirected_dep_table

    def convert_to_word_dependency(self):
        word_lines             = []
        dep_target_indexes     = []
        is_end_of_clause_chars = []
        current_index = 0
        for clause in self.clauses:
            for word_line in clause.word_lines[:-1]:
                word_lines.append(word_line)
                dep_target_indexes.append(current_index + 1)
                is_end_of_clause_chars.append(u"I")
                current_index += 1
            word_lines.append(clause.word_lines[-1])
            if clause.dep_target_index == -1:
                dep_target_index = -1
            else:
                dep_target_clause = self.clauses[clause.dep_target_index]
                dep_target_index = dep_target_clause.begin_index + dep_target_clause.content_word_offset
            dep_target_indexes.append(dep_target_index)
            is_end_of_clause_chars.append(u"E")

        return WdSentence(word_lines, dep_target_indexes, is_end_of_clause_chars,
                          self.file_name, self.sentence_index)

    def to_s(self):
        result_str = u""
        for c in self.clauses:
            result_str += c.to_s()
        result_str += u"EOS\n"
        return result_str

    def info(self):
        return str(self.sentence_index) + " th sentence in " + self.file_name 


class WdSentence:
    def __init__(self, word_lines, dep_target_indexes, is_end_of_clause_chars,
                 file_name=None, sentence_index=None):
        assert \
            len(word_lines)         == len(dep_target_indexes) and \
            len(dep_target_indexes) == len(is_end_of_clause_chars), \
            "arrays of arguments differ in length"

        self.word_n = len(word_lines)
        words = []
        for i in range(self.word_n):
            words.append([word_lines[i],
                          dep_target_indexes[i],
                          is_end_of_clause_chars[i]])
        self.words = words

    def to_s(self):
        result_str = u""
        for i, word in enumerate(self.words):
            result_str += (u"\t".join([str(i), str(word[1]), word[2], word[0]]) + u"\n" )
        result_str += u"EOS\n"
        return result_str


class Clause(object):
    DepPattern = re.compile(r"^\* (\d+) (-?\d+)[A-Z].+$")

    def __init__(self, word_lines, info_line, begin_index):
        self.word_lines = word_lines
        self.info_line = info_line
        self.begin_index = begin_index

        clause_index, dep_target_index = self.__extract_dep_info(info_line)
        if clause_index == dep_target_index:
            raise RuntimeError("dependency target must not be oneself")
        self.clause_index = clause_index
        self.dep_target_index = dep_target_index
        self.content_word_offset = self.__examine_content_word_offset(word_lines)

    def __extract_dep_info(self, info_line):
        match = self.DepPattern.search(info_line)
        if not match:
            raise RuntimeError(
                (u"invalid dependency information line `" + info_line + "`").encode('utf-8'))
        return int(match.group(1)), int(match.group(2))

    def __examine_content_word_offset(self, word_lines):
        for i, word_line in enumerate(reversed(word_lines)):
            if self.__is_content_word(word_line):
                return len(word_lines) - i - 1
        return len(word_lines) - 1 # 内容語が１つもなかったら一番後ろの単語にかける

    def __is_content_word(self, word_line):
        word, feature_str = word_line.split(u"\t")
        features = feature_str.split(u",")
        pos_large = features[0]
        return (pos_large in ContentPOS)

    @classmethod
    def read_from_lines(cls, lines, begin_index):
        info_line = lines[0]
        word_lines = []
        for line in lines[1:]:
            word_lines.append(line)
        return cls(word_lines, info_line, begin_index)

    def to_s(self):
        result_str = u"\n".join([self.info_line] + self.word_lines) + u"\n"
        return result_str


class Environment(object):
    def __init__(self,
                 remove_dummy_dep_sent,   fix_dummy_dep_sent,
                 remove_reverse_dep_sent, remove_non_projective_dep_sent,
                 convert_to_word_dependency):
        if remove_dummy_dep_sent and fix_dummy_dep_sent:
            raise RuntimeError("`remove_dummy_dep_sent` and `fix_dummy_dep_sent` are must not True together")
        self.remove_dummy_dep_sent          = remove_dummy_dep_sent
        self.fix_dummy_dep_sent             = fix_dummy_dep_sent
        self.remove_reverse_dep_sent        = remove_reverse_dep_sent
        self.remove_non_projective_dep_sent = remove_non_projective_dep_sent
        self.convert_to_word_dependency     = convert_to_word_dependency



def main():
    is_target_dir, source, target, env = init_settings()

    if is_target_dir:
        convert_dir(source, target, env)
    else:
        convert_file(source, target, env)


def init_settings():
    parser = OptionParser()
    parser.add_option(
        "-r",
        action="store_true",
        dest="is_target_dir",
        default=False
    )

    parser.add_option(
        "-s", "--source",
        action="store",
        type="str",
        dest="source"
    )

    parser.add_option(
        "-t", "--target",
        action="store",
        type="str",
        dest="target"
    )

    parser.add_option(
        "--dummy",
        action="store",
        dest="dummy",
        choices=["dl", "ch", "kp"],
        default="ch"
    )

    parser.add_option(
        "--reverse",
        action="store_true",
        dest="remove_reverse",
        default=True
    )

    parser.add_option(
        "--non-proj",
        action="store_true",
        dest="remove_non_projective",
        default=True
    )

    parser.add_option(
        "--convert",
        action="store_true",
        dest="convert_to_word_dependency",
        default=False
    )

    options, args = parser.parse_args()

    is_target_dir = options.is_target_dir
    source        = options.source
    target        = options.target

    if options.dummy == "dl":
        remove_dummy = True
        fix_dummy    = False
    elif options.dummy == "ch":
        remove_dummy = False
        fix_dummy    = True
    elif options.dummy == "kp":
        remove_dummy = False
        fix_dummy    = False
    else:
        raise RuntimeError()
    env = Environment(remove_dummy, fix_dummy,
                      options.remove_reverse,
                      options.remove_non_projective,
                      options.convert_to_word_dependency)

    return is_target_dir, source, target, env


def convert_dir(source_dir, target_dir, env):
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        convert_file(source_dir + "/" + file_name,
                     target_dir + "/" + file_name,
                     env)


def convert_file(source_path, target_path, env):
    sentences = read_corpus(source_path)

    if env.remove_dummy_dep_sent:
        _sentences = []
        for sentence in sentences:
            if sentence.has_dummy_dependent():
                print("remove-dummy " + sentence.info())
            else:
                _sentences.append(sentence)
        sentences = _sentences
    else:
        if env.fix_dummy_dep_sent:
            for sentence in sentences:
                if sentence.has_dummy_dependent():
                    sentence.fix_dummy_dependent()
                    print("fix-dummy " + sentence.info())

    if env.remove_reverse_dep_sent:
        _sentences = []
        for sentence in sentences:
            if sentence.has_reverse_dependent():
                print("remove-reverse " + sentence.info())
            else:
                _sentences.append(sentence)
        sentences = _sentences

    if env.remove_non_projective_dep_sent:
        _sentences = []
        for sentence in sentences:
            if not sentence.is_projective():
                print("remove-non-projective " + sentence.info())
            else:
                _sentences.append(sentence)
        sentences = _sentences

    if env.convert_to_word_dependency:
        sentences = [ sentence.convert_to_word_dependency() for sentence in sentences ]

    dump_sentences(sentences, target_path)


def read_corpus(source_path):
    with codecs.open(source_path, "r", "utf-8") as f:
        sentences = []
        sentence_lines = []
        sentence_index = 0
        for line in f:
            line = line.rstrip(u"\n\r")
            if len(line) >= 2 and line[0:2] == u"#!":
                pass
            elif line == u"EOS":
                try:
                    sentence = CdSentence.read_from_lines(sentence_lines,
                                                          source_path, sentence_index)
                    sentences.append(sentence)
                except Exception as ex:
                    sys.stderr.write(str(ex) + "  " + 
                                     str(sentence_index) + " th sentence in " +
                                     source_path + "\n")
                sentence_lines = []
                sentence_index += 1
            else:
                sentence_lines.append(line)
        return sentences


def dump_sentences(sentences, target_path):
    with open(target_path, "w") as f:
        for sentence in sentences:
            f.write(sentence.to_s().encode(u"utf-8"))



if __name__ == "__main__":
    main()
