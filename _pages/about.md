---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am Kenny Wong Lik Hang (王力恒). I received my **BSc in Computer Science** from **City University of Hong Kong (CityUHK)** in 2025 and my **MSc in Computer Science and Engineering** from **The Chinese University of Hong Kong (CUHK)** in 2026.

I am an **MPhil student in Computer Science and Engineering** at **CUHK** (from **1 August 2026 to 31 July 2028**), under the supervision of [**Prof. Dou Qi**](https://www.cse.cuhk.edu.hk/~qdou/). My research interest is in Embodied AI and Robotics.

You can view my CV ([here](/files/resume.pdf)). Feel free to reach out to me at `klhwong3 [at] outlook [dot] com`—I’d love to collaborate on something exciting in the future! :>

<a href="https://visitorbadge.io/status?path=https%3A%2F%2Fkenn3o3.github.io%2F"><img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fkenn3o3.github.io%2F&labelColor=%23d9e3f0&countColor=%232ccce4" /></a>

## News
- (2025-06-04) Graduated from **City University of Hong Kong (CityUHK)** with **BSc in Computer Science**
- (2024-11-01) Admitted to **The Chinese University of Hong Kong (CUHK)** MSc in Computer Science (2025 Fall Entry)

## Project Highlights
Below are some of my research projects:

<table style="width: 100%; border: none;">
  {% assign sorted_projects = site.projects | sort: 'date' | reverse %}
  {% for project in sorted_projects %}
    <tr style="width: 100%; border: none;">
    <td width="20%" style="padding: 10px 30px 10px 10px; border: none;">
      <div class="container">
        <img src="{{ project.thumbnail }}" width="180px" style="box-shadow: 4px 4px 4px #888888; margin-left: 10px;">
      </div>
    </td>
    <td style="padding: 10px 30px 10px 10px; border: none;">
      <a href="{{ project.url }}" style="font-size: 15px; font-weight: bold;">{{ project.title }}</a>
      <div style="height: 5px;"></div>
      <div style="font-size: 12px">{{ project.authors }}</div>
      <div style="height: 5px;"></div>
      <div style="font-size: 12px">{{ project.venue }}, {{ project.date | date: "%Y-%m-%d" }}</div>
      <div style="height: 5px;"></div>
      <div style="font-size: 12px">
        {% if project.paper_url %}<a href="{{ project.paper_url }}" target="_blank">Paper</a> / {% endif %}
        {% if project.poster_url %}<a href="{{ project.poster_url }}" target="_blank">Poster</a> / {% endif %}
        {% if project.code_url %}<a href="{{ project.code_url }}" target="_blank">Code</a> / {% endif %}
        {% if project.arxiv_url %}<a href="{{ project.arxiv_url }}" target="_blank">arXiv</a>{% endif %}
      </div>
    </td>
  </tr>
  {% endfor %}
</table>
