# Predicting Firefighter Interventions in Essonne

[![Build status](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml/badge.svg)](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml)


## Introduction

This challenge involves predicting the number of firefighter interventions in the Essonne department, categorized into five types: 
- **SUAP**: Emergency medical assistance
- **INCN**: Natural fire
- **INCU**: Urban fire
- **ACCI**: Road accident
- **AUTR**: Other

The objective is to build predictive models that can estimate the weekly number of interventions for each category at the municipal level, using historical intervention data and various contextual features, including demographic, meteorological, pollution, and socio-economic factors.


Authors : *Tristan Waddington, Dimitri Iratchet et Fabien Lagnieu*
(IPP / CEMS-T)

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started on this RAMP with the
[dedicated notebook](https://github.com/ramp-kits/map_estimation/blob/main/map_estimation_starting_kit.ipynb).

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
