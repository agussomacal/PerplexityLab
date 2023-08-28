import unittest
from pathlib import Path

from PerplexityLab.LaTexReports import Code2LatexConnector, RunsInfo2Latex


class TestVizUtils(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(__file__).parent.joinpath("TestLatexConnector")
        self.latex_connector = Code2LatexConnector(path, "TestLatexConnector")
        self.latex_connector.create_template()

    def test_RunsInfo2Latex(self):
        runsinfo = RunsInfo2Latex(path2latex=self.latex_connector.path2latex_file)
        runsinfo.append_info(pasta="spaguetti", minutos=8)
        self.latex_connector.add_line("Tipo de pasta: \\perplexityinsert{pasta}.")
        self.latex_connector.add_line("Tiempo de coccion: \\perplexityinsert{minutos}min.")
        self.latex_connector.make_report()
        runsinfo.insert_preamble_in_latex_file()
        self.latex_connector.compile()

    if __name__ == '__main__':
        unittest.main()
