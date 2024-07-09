from moviepy.editor import ImageSequenceClip

def crearVideo():
    clip = ImageSequenceClip('generaciones', fps=1)
    video_filename = 'generations_video.mp4'
    clip.write_videofile(video_filename, codec='libx264')