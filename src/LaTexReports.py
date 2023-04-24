import os
import subprocess
from datetime import date

from src.miscellaneous import check_create_path, clean_str4saving

SECTION_FOLDER_NAME = 'sections'
PLOTS_FOLDER_NAME = 'plots'
GIF_FOLDER_NAME = 'gifs'


class Code2LatexConnector:
    def __init__(self, path, filename):
        self.latex_path = check_create_path(path)
        self.latex_file = filename
        self.main_section = self.latex_file.replace(" ", "_")
        self.text = {self.main_section: ''}
        self.section = self.main_section

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

        self.add_line('\\end{document}', section=self.main_section)

        with open(f'{self.latex_path}/{self.latex_file}.tex', 'w') as f:
            f.write(self.text[self.main_section])

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

    def compile(self):
        cwd = os.getcwd()
        os.chdir(self.latex_path)
        subprocess.run(["pdflatex", "{}.tex".format(self.latex_file)])
        os.chdir(cwd)
