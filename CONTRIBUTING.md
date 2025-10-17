# Contributing to SyGra :telescope:

Thank you for your interest in contributing to SyGra!

This document should be able to guide contributors in their different types of contributions.

:information_source: Just want to ask a question? Open a topic on our [Discussion page](https://github.com/ServiceNow/sygra/discussions).


## Get your environment setup

SyGra is split in two components.
Go to our [Installation](docs/installation.md) to get installation.


[//]: # (It is encouraged to be familiar with our [development best practices]&#40;https://servicenow.github.io/sygra/development/dev-practices/&#41;.)


When everything is set up, you can refer to the

* [Setting up a new Datagen Pipeline](docs/getting_started/create_new_pipeline.md)
* [Graph Configuration Guide](docs/getting_started/graph_config_guide.md)

sections to setup a synthetic data generation pipeline.

## Pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to keep the codebase consistent. Hooks are run automatically on `git commit` (and some on `pre-push`).

### One-time setup

```bash
# Install dependencies (includes pre-commit as a dev dep)
poetry install

# Install Git hooks for this repo
poetry run pre-commit install
poetry run pre-commit install -t pre-push

# (optional) Warm the caches so your first commit is fast
poetry run pre-commit run --all-files
```

#### Why pre-commit?

- Fast feedback on style/format issues
- Consistent code across contributors
- Fewer “nit” comments in PR reviews

## How to submit a bug report

[Open an issue on Github](https://github.com/ServiceNow/sygra/issues/new/choose) and select "Bug report". If you are not sure whether it is a bug or not, submit an issue and we will be able to help you.

Issues with reproducible examples are easier to work with. Do not hesitate to provide your configuration with generated data if need be.

If you are familiar with the codebase, providing a "unit test" is helpful, but not mandatory.

## How to submit changes

First, open an issue describing your desired changes, if it does not exist already. This is especially important if you need to add a new dependency. If that is the case, please mention which package and which version you would like to add. Once a team member accepts your proposition, you can start coding!

You can also self-assign an existing issue by commmenting #self-assign on the issue.

1. [Fork the repo to your own account](https://github.com/ServiceNow/sygra/fork).
2. Clone your fork of the repo locally.
3. Make your changes (the fun part).
4. Commit and push your changes to your fork.
5. [Open a pull-request](https://github.com/ServiceNow/sygra/compare) with your branch.
6. Once a team member approves your changes, we will merge the pull request promptly.

### Guidelines for a good pull-request
When coding, pay special attention to the following:
* Your code should be well commented for non-trivial sections, so it can be easily understood and maintained by others.
* Do not expose any personal/sensitive data.
* Add unit tests when a notable functionality has been added or changed.

[//]: # (* Read our [development best practices]&#40;https://servicenow.github.io/sygra/development/dev-practices/&#41; to set up `pre-commit`, and test your changes.)
* Do not forget to notify the team in advance that you are working on an issue (Using #self-assign or by creating an issue). Mention it if you need to add/bump a dependency.
* Check the [PR template](https://github.com/ServiceNow/sygra/blob/main/.github/pull_request_template.md) in advance to see the checklist of things to do.

### Where to ask for help!

If you need help, feel free to reach out to a team member or through a GitHub Discussion.
If the team member can't answer your question, they will find someone who can!


## Current contributors

- Bidyapati Pradhan [@bidyapati-p](https://github.com/bidyapati-p)
- Surajit Dasgupta [@zephyrzilla](https://github.com/zephyrzilla)
- Amit Kumar Saha [@amitsnow](https://github.com/amitsnow)
- Omkar Anustoop [@omkar-anustoop-ai](https://github.com/omkar-anustoop-ai)
- Sriram Puttagunta [@psriramsnc](https://github.com/psriramsnc)
- Vipul Mittal [@vipul-mittal](https://github.com/vipul-mittal)

We would love to add you to this list!

[//]: # (To reach out to the owners of this project, please see our [About page]&#40;https://servicenow.github.io/sygra/about-us/&#41;.)
