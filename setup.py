from setuptools import setup, find_packages

setup(
    name='email_spam',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'aiohttp==3.8.4',
        'numpy==1.24.1',
        'pandas==1.5.3',
        'scikit-learn==1.2.1',
        'python==3.11',
    ],
)
