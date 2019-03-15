import click

from tqdm import tqdm
from dataset import MXIndexedRecordIO


@click.group()
def main():
    pass


@main.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--strip', default=True)
def build(**kwargs):
    with open(kwargs['input_file']) as f:
        record = MXIndexedRecordIO(f'{kwargs["output_file"]}.idx',
                                   f'{kwargs["output_file"]}.rec', 'w')
        record.open()
        for line in tqdm(f):
            line = line.strip('\n')
            if kwargs['strip']:
                line = ''.join(line.split())
            record.write(line.encode('utf-8'))
        record.close()


if __name__ == '__main__':
    main()
