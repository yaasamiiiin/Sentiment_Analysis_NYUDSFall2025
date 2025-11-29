from setuptools import setup, find_packages

setup(
    name="sentiment_analysis",
    version="1.0.0",
    author="Yasamin",
    description="Sentiment Analysis Data Processing and Modeling Package",
    # This tells setuptools that 'src' is a package
    packages=['src'],
    # This makes src.* modules importable
    py_modules=[
        'src.data_loading',
        'src.data_preprocessing',
        'src.model',
        'src.train'
    ],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "wordcloud",
    ],
    python_requires=">=3.7",
)