
# PlantGL -> PLY
def write(fname, scene):

    """ Write an OBJ file from a plantGL scene graph.
    This method will convert a PlantGL scene graph into an OBJ file.
    It does not manage  materials correctly yet.
    :Examples:
        import openalea.plantgl.scenegraph as sg
        scene = sg.Scene()"""
    import openalea.plantgl as plantgl
    import openalea.plantgl.scenegraph as sg
    import openalea.plantgl.algo as alg

    # print("Write "+fname)
    d = alg.Discretizer()
    f = open(fname, 'w')

    vertices = []  # List of point List
    faces = []  # list  of tuple (offset,index List)

    counter = 0
    for i in scene:
        if i.apply(d):
            p = d.result
            if isinstance(p, plantgl.scenegraph._pglsg.PointSet):
                continue
            pts = p.pointList
            face = p.indexList
            n = len(p.pointList)
            if n > 0:
                color = i.appearance.ambient
                for j in pts:
                    vertices.append((j, color))
                for j in face:
                    j = list(map(lambda x: x + counter, j))
                    faces.append(j)
            counter += n

    header = '''ply
format ascii 1.0
comment author abhinav
comment File Generated with PlantGL 3D Viewer
element vertex {}
property float x
property float y
property float z
property uchar diffuse_red
property uchar diffuse_green
property uchar diffuse_blue
element face {}
property list uchar int vertex_indices 
end_header'''.format(len(vertices), len(faces))
    f.write(header + '\n')
    for pt, color in vertices:
        r, g, b = color
        x, y, z = pt
        f.write('{:.4f} {:.4f} {:.4f} {:.0f} {:.0f} {:.0f}\n'.format(x, y, z, r, g, b))
    for face in faces:
        f.write('{:.0f}'.format(len(face)))
        for a in face:
            f.write(' {}'.format(a))
        f.write('\n')

    f.close()


def convert_ply_to_obj(in_path, out_path):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_path)
    ms.save_current_mesh(out_path)
    print("Done")
