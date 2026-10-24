# Personal Website

Gemfile:
```
source 'https://rubygems.org'

group :jekyll_plugins do
  gem 'jekyll'
  gem 'jekyll-feed'
  gem 'jekyll-sitemap'
  gem 'jemoji'
  gem 'webrick', '~> 1.8'
end

gem 'github-pages'
```

`jekyll serve -l -H localhost`

Template adapted from https://academicpages.github.io/



For windows:

1. Go to [RubyInstaller for Windows](https://rubyinstaller.org/) and download the Ruby+Devkit version (e.g., “Ruby x.x.x-x (x64) with Devkit”).

2. Run the installer. When prompted, select the option to install the MSYS2 development toolchain (this is required to build certain Ruby gems).

```
gem install bundler
gem install jekyll
jekyll -v
```

Gemfile:
```
source 'https://rubygems.org'

gem 'github-pages', group: :jekyll_plugins
gem 'tzinfo-data', platforms: [:mingw, :mswin, :x64_mingw, :jruby]
gem 'webrick', '~> 1.8'
```

```
bundle install
bundle exec jekyll serve
```

Note: Modify

_pages\about.md
_config.yml (marked #change here)
add a _pages/(projects).md