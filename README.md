<<<<<<< HEAD
# AI Circuit



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.garage.epam.com/ai-circuit/ai-circuit.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://git.garage.epam.com/ai-circuit/ai-circuit/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
=======
# H&M Fashion Classification — Agentic ML Loop

5-class fashion image classification on the H&M dataset, used as a benchmark to demonstrate an **autonomous AI ML engineer** that iteratively improves model performance without human intervention.

## Project Goal

This is **not** a model accuracy competition. The objective is to show an agentic experimentation loop where an AI agent:
1. Runs a training experiment
2. Reads the results (metrics, per-class F1, confusion matrix)
3. Writes experiment notes
4. Proposes config changes
5. Repeats until a target F1 is reached or iterations are exhausted

See [Project.md](Project.md) for the full system architecture.

---

## Setup

### Requirements

```bash
pip install torch torchvision timm scikit-learn tensorboard pyyaml langgraph openai
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
```

### Directory Structure

```
H&M_data/
├── data/
│   ├── sample/          # 500 train / 63 val / 63 test per class (fast iteration)
│   ├── full/            # full dataset (symlink → processed_data/)
│   ├── class_weights.json
│   └── class_mapping.json
├── agents/
│   ├── hm_training_agent.py   # LangGraph agent
│   └── prompts.py             # LLM prompt templates
├── experiments/               # auto-created; one subdir per agent run
│   ├── run_1/
│   │   ├── config.yaml        # exact config used
│   │   ├── best_model.pth     # best checkpoint
│   │   ├── metrics.json       # full metrics + history
│   │   ├── notes.md           # LLM-written experiment notes
│   │   └── tensorboard/
│   └── experiment_log.json    # cross-run summary
├── processed_data/            # original preprocessed splits (full scale)
├── train.py                   # training pipeline (config-driven, agent does not modify)
├── training_config.yaml       # baseline config (agent reads this as starting point)
├── run_agent.py               # agent entry point
└── create_sample_data.py      # creates data/sample/ from processed_data/
```

---

## Quick Start

### Run the agentic loop (sample data, 5 iterations)

```bash
python run_agent.py --max-iterations 5 --target-f1 0.75
```

### Run a single training manually

```bash
python train.py --config training_config.yaml
```

### Switch to full dataset

Edit `training_config.yaml`:
```yaml
paths:
  data_dir: data/full
training:
  max_samples_per_class: null   # or set e.g. 2000 for capped full-data runs
```

### Monitor training

```bash
tensorboard --logdir experiments/
```

---

## Config Reference

Key fields the agent is allowed to modify:

| Key | Default | Options |
|-----|---------|---------|
| `model.backbone` | `efficientnet_b0` | `efficientnet_b0/b2/b4`, `resnet18/50`, `convnext_tiny` |
| `model.checkpoint` | `null` | path to `.pth` for warm-starting |
| `optimizer.lr` | `0.0003` | 1e-5 to 5e-4 |
| `optimizer.type` | `adamw` | `adamw`, `adam`, `sgd` |
| `scheduler.type` | `cosine` | `cosine`, `step`, `onecycle`, `plateau` |
| `loss.type` | `weighted_ce` | `weighted_ce`, `focal` |
| `augmentations.mixup` | `false` | `true/false` |
| `augmentations.cutmix` | `false` | `true/false` |
| `augmentations.randaugment` | `false` | `true/false` |
| `training.max_samples_per_class` | `null` | integer or `null` |

---

## Dataset

H&M Personalized Fashion — 5 classes from `product_group_name`:

| Class | Train (full) | Weight |
|-------|-------------|--------|
| Garment Upper body | 34,144 | 0.43 |
| Garment Lower body | 15,816 | 0.93 |
| Garment Full body | 10,620 | 1.38 |
| Accessories | 8,804 | 1.67 |
| Shoes | 4,125 | 3.56 |

Class weights are inverse-frequency normalized for `WeightedCrossEntropy` and `WeightedRandomSampler`.

---

## Workflows

**Workflow 1 (this repo):** Human prepares data → AI agent trains and iterates  
**Workflow 2:** Agent receives raw files only → decides data structure → trains and iterates

See [Project.md](Project.md) for both workflows in detail.
>>>>>>> 8353d3a (initial commit)
