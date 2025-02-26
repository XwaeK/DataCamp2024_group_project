# Non-invasive estimation of the MAP using PPG

[![Build status](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml/badge.svg)](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml)

## Introduction

The continuous monitoring of blood pressure is a major challenge in the general anesthesia field. Indeed, the monitoring of blood pressure is essential to ensure that the patient is stable during the operation. However, the current methods to measure blood pressure are either non-invasive, but non-continuous, or invasive, which can lead to complications, and expensive, making them inadapted in many situations.
A potential solution to this problem is to use non-invasive monitoring signals which are routinely collected, like the electrocardiogram (ECG)  photoplethysmogram (PPG) signal, to estimate the mean arterial pressure (MAP) using AI. The PPG signal is a non-invasive signal that can be easily acquired using a pulse oximeter. The MAP is a measure of the average blood pressure in an individual's arteries during one cardiac cycle.
The goal of this challenge is to estimate the MAP from the non-invasive signals.

Authors : Thomas Moreau (Inria), François Caud (DATAIA - Université Paris-Saclay), Jade Perdereau (APHP, Inria)

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
