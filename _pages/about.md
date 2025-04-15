---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a senior year undergraduate student majoring in Computer Science, my research interests lie in Deep Learning, 3D Computer Vision, Embodied AI, Robotics, Continual Learning. 
<!-- My [CV](https://Kenn3o3.github.io/files/resume.pdf) -->

Please reach out to me via `klhwong3 [at] outlook [dot] com`. I am interested in working on something together in the future :>

## News
- Currently grinding
- (2024/11/01) Admitted to **CUHK MSc CS (2025 Fall Entry)**

## Project Highlights
Here are some of my featured projects:

<table style="width: 100%; border: none;"> {% for project in site.projects %} <tr style="width: 100%; border: none;"> <td width="20%" style="padding: 10px 30px 10px 10px; border: none;"> <div class="container"> <img src="{{ project.thumbnail }}" width="180px" style="box-shadow: 4px 4px 4px #888888; margin-left: 10px;"> </div> </td> <td style="padding: 10px 30px 10px 10px; border: none;"> <a href="{{ project.url }}" style="font-size: 15px; font-weight: bold;">{{ project.title }}</a> <div style="height: 5px;"></div> <div style="font-size: 12px">{{ project.authors }}</div> <div style="height: 5px;"></div> <div style="font-size: 12px">{{ project.venue }}</div> <div style="height: 5px;"></div> <div style="font-size: 12px"> {% if project.paper_url %}<a href="{{ project.paper_url }}" target="_blank">Paper</a> / {% endif %} {% if project.poster_url %}<a href="{{ project.poster_url }}" target="_blank">Poster</a> / {% endif %} {% if project.code_url %}<a href="{{ project.code_url }}" target="_blank">Code</a> / {% endif %} {% if project.arxiv_url %}<a href="{{ project.arxiv_url }}" target="_blank">arXiv</a>{% endif %} </div> <div style="height: 5px;"></div></td> </tr> {% endfor %} </table>