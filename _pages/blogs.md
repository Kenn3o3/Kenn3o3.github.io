---
layout: archive
title: "Blogs"
permalink: /blogs/
---

<table style="width: 100%; border: none;">
  {% assign sorted_blogs = site.blogs | sort: 'date' | reverse %}
  {% for blog in sorted_blogs %}
    <tr>
      <td style="border: none;">
        <a href="{{ blog.url }}" style="font-size: 15px; font-weight: bold;">{{ blog.title }} ({{ blog.date | date: "%Y-%m-%d" }})</a>
        <div style="height: 3px;"></div>
      </td>
    </tr>
  {% endfor %}
</table>