import io
import zipfile

# from core.utils.export_filename import generate_svg_filename
# # from core.utils.svg_export import export_svg_file


# class SvgGenerationJob:
#     def __init__(self, contours: list[dict], output_dir: str, basename: str):
#         """
#         Initializes the SVG generation job.
#         Args:
#             contours: A list of contour dictionaries (each with 'elevation', 'geometry', etc.).
#             output_dir: Directory to write the SVG files into.
#             basename: Base filename to use for each layer's SVG.
#         """
#         self.contours = contours
#         self.output_dir = output_dir
#         self.basename = basename

#     def run(self) -> list[str]:
#         """
#         Generates SVG files from the provided contours and writes them to disk.

#         Returns:
#             List of paths to the generated SVG files.
#         """
#         filenames = []
#         for i, layer in enumerate(self.contours):
#             filename = generate_svg_filename(i, layer["elevation"], self.basename)
#             svg_path = f"{self.output_dir}/{filename}"
#             export_svg_file(
#                 svg_path, layer["geometry"], layer.get("text", ""), layer["thickness"]
#             )
#             filenames.append(svg_path)
#         return filenames


# class ZipExportJob:
#     def __init__(self, file_paths: list[str]):
#         """
#         Initializes a ZIP packaging job for given file paths.

#         Args:
#             file_paths: List of file paths to include in the ZIP archive.
#         """
#         self.file_paths = file_paths

#     def run(self) -> bytes:
#         """
#         Creates an in-memory ZIP archive containing the specified files.

#         Args:
#             None
#         Returns:
#             bytes: The contents of the ZIP archive.
#         """
#         zip_buffer = io.BytesIO()
#         with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
#             for path in self.file_paths:
#                 arcname = path.split("/")[-1]
#                 zip_file.write(path, arcname)
#         zip_buffer.seek(0)
#         return zip_buffer.read()
