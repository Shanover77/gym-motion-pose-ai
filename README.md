gym-motion-pose-ai: An on-going project to critique an exercise by using an ensemble of ML/Vision models. Mainly focuses on orientations, angle of joints, based on the human pose estimate (33 Joints)

## Key Milestones
1. Repetition detection model
2. Orientation and symmetry model
3. Threshold predictor for training step

## Notebooks
Contains experimental notebooks that has the code for ETL/EDA.

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`VIDEO_FILE_NAME=barbell_row.mp4`


## Run Locally

Clone the project

```bash
  git clone "repository url"
```

Go to the project directory

```bash
  cd gym-motion-pose-ai
```

Create a Virtual Environment
```bash
  python3.11 -m venv gym-motion-pose-aienv
```

Activate Virtual Environment
```bash
  source gym-motion-pose-aienv/bin/activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python main.py
```

# Contributors

- [@ujjawalpoudel](https://github.com/ujjawalpoudel)
- [@Shanover77](https://github.com/Shanover77)
- [@Sunilrai486](https://github.com/Sunilrai486)
- [@Bigyan835](https://github.com/Bigyan835)
- [@ANJAISANILKUMAR](https://github.com/ANJAISANILKUMAR)
- [@syedzumairh](https://github.com/syedzumairh)

## References

Mihai Fieraru, Mihai Zanfir, Silviu-Cristian Pirlea, Vlad Olaru, and Cristian Sminchisescu.  
"AIFit: Automatic 3D Human-Interpretable Feedback Models for Fitness Training."  
In *The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2021.  
[Link to the paper](https://openaccess.thecvf.com/content/CVPR2021/html/Fieraru_AIFit_Automatic_3D_Human-Interpretable_Feedback_Models_for_Fitness_Training_CVPR_2021_paper.html)

