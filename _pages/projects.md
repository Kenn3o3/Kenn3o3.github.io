---
layout: archive
title: "Projects"
permalink: /projects/
---

<table class="project-highlights">
  {% assign sorted_projects = site.projects | sort: 'highlight_order' | reverse %}
  {% for project in sorted_projects %}
    <tr class="project-row">
      <td class="project-thumbnail-cell">
        <div class="project-thumbnail">
          <img src="{{ project.thumbnail }}" class="project-thumbnail-image" alt="">
        </div>
      </td>
      <td class="project-details">
        <a href="{{ project.url }}" class="project-title">{{ project.title }}</a>
        <div class="project-authors">{{ project.authors }}</div>
        <div class="project-meta"><em class="project-venue">{{ project.venue }}</em><span class="project-date"> · {{ project.date | date: "%Y-%m-%d" }}</span></div>
        <div class="project-links">
          {% if project.project_url %}<a href="{{ project.project_url }}" target="_blank">Project Page</a>{% endif %}
          {% if project.paper_url %}<a href="{{ project.paper_url }}" target="_blank">Paper</a>{% endif %}
          {% if project.poster_url %}<a href="{{ project.poster_url }}" target="_blank">Poster</a>{% endif %}
          {% if project.code_url %}<a href="{{ project.code_url }}" target="_blank">Code</a>{% endif %}
          {% if project.arxiv_url %}<a href="{{ project.arxiv_url }}" target="_blank">arXiv</a>{% endif %}
        </div>
        <div class="project-excerpt">{{ project.excerpt }}</div>
      </td>
    </tr>
  {% endfor %}
</table>
