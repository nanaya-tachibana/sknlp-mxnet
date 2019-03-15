from setuptools import setup, find_packages

setup(
    name='elmo',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'tqdm'
    ],
    entry_points='''
        [console_scripts]
        model-cli=create_dataset:main
    ''',
)
