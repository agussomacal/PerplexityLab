import os
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from PerplexityLab.miscellaneous import check_create_path, clean_str4saving

SECTION_FOLDER_NAME = 'sections'
PLOTS_FOLDER_NAME = 'plots'
GIF_FOLDER_NAME = 'gifs'


class RunsInfo2Latex:
    """
    Utility to insert info from runs directly into the latex via the latex command \perplexityinsert automatically
    added to the laTex file preamble through insert_preamble_in_latex_file. Information is extracted from file
    runsinfo.csv whose info is added in the python code via append_info.
    """
    def __init__(self, path2latex):
        self.path2latex = Path(path2latex)
        self.runs_info_filepath = Path.joinpath(self.path2latex.parent, "runsinfo.csv")

    def append_info(self, **kwargs):
        data = pd.Series(pd.read_csv(self.runs_info_filepath, names=["thekey", "thevalue"], index_col=0)) \
            if os.path.exists(self.runs_info_filepath) else pd.Series()
        for k, v in kwargs.items():
            data[k] = v

        data.to_csv(self.runs_info_filepath, header=False)

    def insert_preamble_in_latex_file(self):
        with open(self.path2latex, "r") as f:
            contents = f.readlines()

        index = 0
        for i, line in enumerate(contents):
            if "\\usepackage" in line:
                index = i + 1
            elif "Perplexity" in line:
                print(f"Preamble already in LaTex file {self.path2latex}. Aborting insertion.")
                return

        value = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" \
                "%              Perplexity inserter preamble                %\n" \
                "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" \
                "% Import packages and define command \insertval for inputing experiments information directly \n" \
                "% into the latex to keep an updated-paired version of the article.\n\n" \
                "% package to open file containing variables\n" \
                "\\usepackage{datatool, filecontents}\n" \
                "\DTLsetseparator{,}% Set the separator between the columns.\n" \
                "\DTLloadrawdb[noheader, keys={thekey,thevalue}]{runsinfo}{runsinfo.csv}\n" \
                "% % import data\n" \
                "% % Loads mydata.dat with column headers 'thekey' and 'thevalue'\n" \
                "% \\newcommand{\perplexityinsert}[1]{\DTLfetch{runsinfo}{thekey}{#1}{thevalue}}\n" \
                "\\newcommand{\perplexityinsert}[1]{\DTLgetvalueforkey{\datavalue}{thevalue}{runsinfo}{thekey}{#1}\datavalue}\n" \
                "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" \
                "%                End Perplexity inserter                   %\n" \
                "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        contents.insert(index, value)

        with open(self.path2latex, "w") as f:
            contents = "".join(contents)
            f.write(contents)
        print(f"Perplexity preamble added to LaTex file {self.path2latex}.")


class Code2LatexConnector:
    def __init__(self, path, filename):
        self.latex_path = check_create_path(path)
        self.latex_file = filename + (".tex" if filename[-4:] != ".tex" else "")
        self.main_section = filename.replace(" ", "_").split(".")[0]
        self.text = {self.main_section: ''}
        self.section = self.main_section

    @property
    def path2latex_file(self):
        return self.latex_path.joinpath(self.latex_file)

    def create_template(self):
        self.add_line('\\documentclass{article}')
        self.add_line()
        self.add_line('\\usepackage{graphicx}')

        self.add_line('\\usepackage[utf8]{inputenc}')
        self.add_line('\\usepackage{hyperref}')
        self.add_line('\\usepackage{animate}')
        self.add_line('\\usepackage{float}')
        self.add_line('\\usepackage{mathtools}')
        self.add_line('\\usepackage{listings}')
        self.add_line('\\usepackage{color}')

        self.add_line()
        self.add_line()

        self.add_line('\\definecolor{dkgreen}{rgb}{0, 0.6, 0}')
        self.add_line('\\definecolor{gray}{rgb}{0.5, 0.5, 0.5}')
        self.add_line('\\definecolor{mauve}{rgb}{0.58, 0, 0.82}')

        self.add_line('\\lstset{frame = tb, language = Python, aboveskip = 3 mm, belowskip = 3 mm,' + \
                      'showstringspaces = false, columns = flexible, basicstyle = {\\small\\ttfamily}, numbers = none,' + \
                      'numberstyle =\\tiny\\color{gray}, keywordstyle =\\color{blue}, commentstyle =\\color{dkgreen}, ' + \
                      'stringstyle =\\color{mauve}, breaklines = true, breakatwhitespace = true, tabsize = 3}')

        self.add_line()
        self.add_line()

        self.add_line('\\title{' + str(self.latex_file.split(".")[0]) + '}')
        self.add_line('\\author{' + str() + '}')
        self.add_line('\\date{' + str(date.today()) + '}')

        self.add_line()
        self.add_line()
        self.add_line('\\begin{document}')

        self.add_line()
        self.add_line('\\maketitle')
        self.add_line('% \\tableofcontents')
        self.add_line('\\newpage')

        # self.add_line('\\end{document}', section=self.main_section)
        #
        # with open(f'{self.path2latex_file}', 'w') as f:
        #     f.write(self.text[self.main_section])

    def add_line(self, line='', section=None):
        self.text[self.section if section is None else section] += line + '\n'

    def get_plot_path(self, section=None):
        path = check_create_path(self.latex_path, "figures")
        return path if section is None else check_create_path(path, section)

    def includegraphics(self, plot_path):
        # print('%init -> {}'.format(hash(relative_path2copy)))
        print('\\begin{figure}[H]')
        print(
            '\\centerline{\\includegraphics[width=1' + '\\textwidth]' + '{' + plot_path + '}}')
        print('\\caption{}')
        print('\\label{fig:' + clean_str4saving(plot_path.split("/")[-1]) + '}')
        print('\\end{figure}')
        # print('%end -> {}'.format(hash(relative_path2copy)))

    @staticmethod
    def section_name2save(section_name, with_tex=True):
        return section_name.replace(' ', '_') + ('.tex' if with_tex else '')

    def make_report(self):
        for section_name in self.text.keys():
            if section_name == self.main_section:
                path2section = check_create_path(self.latex_path)
                self.add_line('\\end{document}', section=self.main_section)
            else:
                path2section = check_create_path(self.latex_path, SECTION_FOLDER_NAME)
            with open(f"{path2section}/{self.section_name2save(section_name)}", 'w') as f:
                f.write(self.text[section_name])

    def compile(self):
        cwd = os.getcwd()
        os.chdir(self.latex_path)
        subprocess.run(["pdflatex", "{}".format(self.latex_file)])
        os.chdir(cwd)
