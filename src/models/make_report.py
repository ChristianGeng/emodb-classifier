import papermill as pm
import nbformat
# import time

from urllib.request import urlopen
import tempfile
from traitlets.config import Config
from nbconvert import HTMLExporter

import argparse
import sys
# from IPython.display import Javascript
from traitlets import Integer
from nbconvert.preprocessors import Preprocessor


def run_notebook(tmpfile, clfname='GMM_basic'):
    # clfname = 'svm_basic'

    pm.execute_notebook(
        'report_nb.ipynb',
        tmpfile,
        parameters=dict(clf_name=clfname),
        progress_bar=False,
        log_output=False,
        report_mode=False,
        request_save_on_cell_execute=False
    )


class PelicanSubCell(Preprocessor):
    """A Pelican specific preprocessor to remove some of the cells of a notebook"""

    # I could also read the cells from nb.metadata.pelican if someone wrote a JS extension,
    # but for now I'll stay with configurable value.
    start = Integer(6,  help="first cell of notebook to be converted")
    end = Integer(9, help="last cell of notebook to be converted")
    start.tag(config='True')
    end.tag(config='True')

    def preprocess(self, nb, resources):
        self.log.info("I'll keep only cells from %d to %d", self.start, self.end)
        nb.cells = nb.cells[self.start:self.end]
        return nb, resources


def save_notebook():
    # from IPython.display import Javascript
    # display(
    #     Javascript("IPython.notebook.save_notebook()"),
    #     include=['application/javascript']
    # )
    pass

# def output_HTML(exporter, output_file, output):
    # exporter = HTMLExporter()
    # read_file is '.ipynb', output_file is '.html'
    # output_notebook = nbformat.read(read_file, as_version=4)
    # output, resources = exporter.from_notebook_node(output_notebook)


def output_HTML(read_file, output_file='tmp.html'):
    import codecs

    c = Config()
    # c.PelicanSubCell.start = 7
    # c.PelicanSubCell.end = 8
    # c.HTMLExporter.preprocessors = ['nbconvert.preprocessors.ExtractOutputPreprocessor', PelicanSubCell]
    # c.RSTExporter.preprocessors = [PelicanSubCell]
    c.HTMLExporter.preprocessors = [PelicanSubCell]
    exporter = HTMLExporter(config=c)
    # exporter.template_file = 'basic'

    # 3. Process the notebook we loaded earlier
    output, resources = exporter.from_notebook_node(read_file)
    codecs.open(output_file, 'w', encoding='utf-8').write(output)

def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Write Fib Report")
    parser.add_argument(
        dest="clf",
        type=str,
        help="name of classifier")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    tmpfile = tempfile.mktemp(suffix='.ipynb')
    run_notebook(tmpfile, clfname=args.clf)
    url = 'file://'+tmpfile
    response = urlopen(url).read().decode()
    jake_notebook = nbformat.reads(response, as_version=4)

    # save_notebook()

    # save_notebook()
    # time.sleep(2)
    # current_file = 'GMM.ipynb'
    # output_file = 'output_file.html'
    output_HTML(jake_notebook, output_file=args.clf+'.html')

    #     exporter = HTMLExporter()
    #     # read_file is '.ipynb', output_file is '.html'
    #     output_notebook = nbformat.read(read_file, as_version=4)
    #     output, resources = exporter.from_notebook_node(output_notebook)
    #     codecs.open(output_file, 'w', encoding='utf-8').write(output)

    # codecs.open(output_file, 'w', encoding='utf-8').write(body)

if __name__ == '__main__':
    main(sys.argv[1:])
