from moviepy.editor import ImageSequenceClip

# Crear un video a partir de las imágenes
clip = ImageSequenceClip('gen_images', fps=1)  # fps controla la velocidad del video
video_filename = 'generations_video.mp4'
clip.write_videofile(video_filename, codec='libx264')