# lightning-ai-deepl-course
Following along with the Lightning AI Deep Learning Fundamentals [course](https://lightning.ai/courses/deep-learning-fundamentals/)

- [Github repo for course](https://github.com/Lightning-AI/dl-fundamentals)

## Setup

- Install Python >= 3.10. I usually use `pyenv` for managing multiple python versions (see this [nice tutorial](https://realpython.com/intro-to-pyenv/#installing-pyenv) from Real Python that I refer to a lot if you've never used pyenv before)

- Install poetry (I usually use the [official installer](https://python-poetry.org/docs/#installing-with-the-official-installer) method)

- Install the packages in a virtualenv and activate it
```bash
poetry config virtualenvs.in-project true
poetry install --no-root
source .venv/bin/activate
```



