---
layout: archive
title: "Blogs"
permalink: /blogs/
---

<ul class="post-index">
  {% assign sorted_blogs = site.blogs | sort: 'date' | reverse %}
  {% for blog in sorted_blogs %}
    <li>
      <a href="{{ blog.url | relative_url }}">{{ blog.title }}</a>
      <time datetime="{{ blog.date | date: '%Y-%m-%d' }}">{{ blog.date | date: "%b %Y" }}</time>
    </li>
  {% endfor %}
</ul>
