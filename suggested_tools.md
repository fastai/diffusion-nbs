# Tools and Notebooks

The AI art community has been sharing notebooks and making tools to use these models at an ever-increasing rate, to the point where lists of available options [like this one from @pharmapsychotic](https://pharmapsychotic.com/tools.html) can be fairly overwhelming!
So, here are a chosen few that you might want to check out as a starting point.

## Easy online tools

[Dreamstudio](https://beta.dreamstudio.ai/) is a fast and easy online interface for stable diffusion. You can do ~200 generations for free and then credits cost about $0.01 per image. There is also an API which let's you use their servers for inference, which is what powers things like [this photoshop plugin](https://exchange.adobe.com/apps/cc/114117da/stable-diffusion) for quickly generating images without a local GPU.

[Artbreeder's new beta collage feature](https://www.artbreeder.com/beta/collage) is a fun (and free) interface for image-to-image style generations. 

There are a number of other online tools (midjourney, Wombo, NighCafe), typically with some sort of subscription or membership needed. And if you don't mind the occasional wait there are also [spaces on HuggingFace](https://huggingface.co/spaces/stabilityai/stable-diffusion) that remain free and typically keep the interface simple.

## Fancier Interfaces

People in the community have rushed to create new interfaces with additional advanced options for generation. For example, [this stable diffusion webui](https://github.com/sd-webui/stable-diffusion-webui) by hlky has a choice of several slick interfaces.
It has tabs for multiple tasks, batch functionality, integration of things like textual inversion, support for managing a number of model versions and much more. Settings are saved in the metadata of the resulting images, making it easy to access or share your favourite presets. 
This is great for running SD locally, but there is also a [colab notebook](https://colab.research.google.com/github/altryne/sd-webui-colab/blob/main/Stable_Diffusion_WebUi_Altryne.ipynb) if you'd prefer to use Google's GPUs. The [InvokeAI Stable Diffusion toolkit](https://invoke-ai.github.io/InvokeAI/) is another new alternative worth exploring, and features a number of tools as well as optimizations for running locally with as little as 4GB RAM.

The other notable mention in terms of local interfaces is Softology's ['Visions of Chaos'](https://softology.pro/voc.htm) tool, which makes it fairly easy to run a stagering number different notebooks/pipelines locally on Windows.

## Colab Notebooks

There are hundreds of notebooks floating around adding different bits of functionality, but two of the most popular and fully-featured versions are

[Deforum](https://deforum.github.io/) - a community-built notebook with lots of features (and many more planned) including settings for animations and video. [Browsing Twitter for results](https://twitter.com/search?q=%23DeforumDiffusion&src=typeahead_click) will show the kinds of trippy animation this can make - by including a depth estimation model in the pipeline they've made it easy to get all sorts of interesting perspective warping motion into animations.

[Disco Diffusion](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb). Before we had Stable Diffusion, we had to use CLIP to guide unconditional diffusion models. This notebook shows how much you can do with that concept, producing amazing imagery and animations by steering the generation of an unconditional 512x512px diffusion model. 

## Keeping up with the space

News tools appear daily. Rather than chasing every new thing, I recommend spending some time learning a couple of tools well.
YouTube channels [like this one](https://www.youtube.com/c/NerdyRodent/videos) are a good place to see new things in action and evaluate whether you really need that new feature.
Many tools also have a community where you can get involved sharing feedback or helping to add new features - keep an eye out for Discord links and get involved :)

