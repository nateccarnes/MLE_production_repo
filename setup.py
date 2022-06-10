from distutils.core import setup

def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert_text('README.md', 'rst', 'md')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()

setup(
    name='Capstone',
    version='1.0',
    description='Analysis of Capstone Project',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    # Substitute <github_account> with the name of your GitHub account
    url='https://github.com/nateccarnes/MLE_production_repo',
    author='Nathan Carnes',
    author_email='nateccarnes@gmail.com',
    license='None',
    packages=['Capstone'],
    install_requires=[
        'pypandoc>=1.8.1',
        'pandas>=1.4.2',
        'scikit-learn>=1.0.2',
        'numpy>=1.22.3',
        'tensorflow>=2.9.1',
        'keras>=2.4.3'
    ],
)