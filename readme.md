gym-motion-pose-ai: An on-going project to critique an exercise by using an ensemble of ML/Vision models. Mainly focuses on orientations, angle of joints, based on the human pose estimate (33 Joints)

## Key Milestones
1. Repetition detection model - client/src/preprocessor_videos.py => step6_applyPeakValley()
2. Orientation/Symmetry - (pending) - /translation_angle
3. Threshold predictor for training step - (pending) - /threshold

# Client - Client and Trainer

Dir : client/
Client Application (Windows Exec) : client/app.py
Preprocessor videos (Requires /videos/{label}/**.mp4) : client/preprocessor_videos.py -> /trainable_data
Trainer on videos (Requires /trainable_data/*.csv) : client/trainer.py -> /temp/

# Server - RabbitMQ/Flask for Inference
Dir : server/
Listens on flask, requires RabbitMQ and Erlang. See /server/readme.md

# Challenges/Limitations:
1. 2D and 3D is big challenge. We can only get so much information from 2D mediapipe representation.
2. 'Non-full-body' videos or frames may produce undesirable results

# Contributors

- [@ujjawalpoudel](https://github.com/ujjawalpoudel)
- [@Shanover77](https://github.com/Shanover77)
- [@Sunilrai486](https://github.com/Sunilrai486)
- [@Bigyan835](https://github.com/Bigyan835)
- [@ANJAISANILKUMAR](https://github.com/ANJAISANILKUMAR)
- [@syedzumairh](https://github.com/syedzumairh)

# Dataset Used (with thanks) to
INSTITUTE OF MATHEMATICS "SIMION STOILOW" OF THE ROMANIAN ACADEMY 
https://fit3d.imar.ro/

## References/Related works

Mihai Fieraru, Mihai Zanfir, Silviu-Cristian Pirlea, Vlad Olaru, and Cristian Sminchisescu.  
"AIFit: Automatic 3D Human-Interpretable Feedback Models for Fitness Training."  
In *The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2021.  
[Link to the paper](https://openaccess.thecvf.com/content/CVPR2021/html/Fieraru_AIFit_Automatic_3D_Human-Interpretable_Feedback_Models_for_Fitness_Training_CVPR_2021_paper.html)

