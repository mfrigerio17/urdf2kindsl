#!/usr/bin/env python3
'''
Created on August 2018

@author: Marco Frigerio
'''
from __future__ import print_function
import logging, sys, argparse, math
import xml.etree.ElementTree as ET
from collections import OrderedDict as ODict
import numpy as np


logger = None


'''
Extrinsic rotations are about the axes of the original coordinate system, which
is assumed to remain motionless. This is the convention of the 'rpy' attribute
of the URDF format.
Intrinsic rotations are about the axes of a rotating coordinate system, attached
to the moving body, which changes its orientation after each individual
rotation. This is the convention of that 'rotation' attribute of the RobCoGen
format.
'''


def getR_intrinsicXYZ(rx, ry, rz):
    '''
    This is the rotation matrix **base_R_rotated**, where 'rotated' is obtained
    from 'base' with the **intrinsic** rotations rx, ry, and rz

                 cos(ry) cos(rz)                             - cos(ry) sin(rz)                     sin(ry)
    cos(rx) sin(rz) + sin(rx) sin(ry) cos(rz)    cos(rx) cos(rz) - sin(rx) sin(ry) sin(rz)    - sin(rx) cos(ry)
    sin(rx) sin(rz) - cos(rx) sin(ry) cos(rz)    cos(rx) sin(ry) sin(rz) + sin(rx) cos(rz)     cos(rx) cos(ry)
    '''
    sx = math.sin(rx)
    cx = math.cos(rx)
    sy = math.sin(ry)
    cy = math.cos(ry)
    sz = math.sin(rz)
    cz = math.cos(rz)
    return np.array(
        [ [cy*cz            , - cy*sz          ,sy      ],
          [cx*sz + cz*sx*sy , cx*cz - sx*sy*sz , - cy*sx],
          [sx*sz - cx*cz*sy , cx*sy*sz + cz*sx ,  cx*cy ] ] )

'''
Extract the intrinsic Euler angles XYZ from the rotation matrix **base_R_rotated**
'''
def getIntrinsicXYZFromR( R ) :
    rx = math.atan2(-R[1,2], R[2,2])
    ry = math.asin( R[0,2] )
    rz = math.atan2(-R[0,1], R[0,0])

    return (rx, ry, rz)


def getR_extrinsicXYZ(rx, ry, rz):
    '''
    This is the rotation matrix **base_R_rotated**, where 'rotated' is obtained from
    'base' with the **extrinsic** rotations rx, ry, and rz

    cos(ry) cos(rz)    sin(rx) sin(ry) cos(rz) - cos(rx) sin(rz)    sin(rx) sin(rz) + cos(rx) sin(ry) cos(rz)
    cos(ry) sin(rz)    sin(rx) sin(ry) sin(rz) + cos(rx) cos(rz)    cos(rx) sin(ry) sin(rz) - sin(rx) cos(rz)
       - sin(ry)                    sin(rx) cos(ry)                              cos(rx) cos(ry)
    '''
    sx = math.sin(rx)
    cx = math.cos(rx)
    sy = math.sin(ry)
    cy = math.cos(ry)
    sz = math.sin(rz)
    cz = math.cos(rz)
    return np.array(
        [[cy*cz,  cz*sx*sy - cx*sz,  sx*sz + cx*cz*sy],
         [cy*sz,  sx*sy*sz + cx*cz,  cx*sy*sz - cz*sx],
         [ - sy,        cy*sx     ,        cx*cy      ]] )


def _extrinsic2intrinsic_XYZ(erx, ery, erz):
    sx = math.sin(erx)
    cx = math.cos(erx)
    sy = math.sin(ery)
    cy = math.cos(ery)
    sz = math.sin(erz)
    cz = math.cos(erz)

    irx = math.atan2( sx*cz-cx*sy*sz, cx*cy)
    iry = math.asin ( sx*sz + cx*sy*cz )
    irz = math.atan2( cx*sz - sx*sy*cz, cy*cz )

    return (irx, iry, irz)

def _intrinsic2extrinsic_XYZ(irx, iry, irz):
    sx = math.sin(irx)
    cx = math.cos(irx)
    sy = math.sin(iry)
    cy = math.cos(iry)
    sz = math.sin(irz)
    cz = math.cos(irz)

    erx = math.atan2(cx*sy*sz + sx*cz, cx*cy)
    ery = math.asin(cx*sy*cz - sx*sz)
    erz = math.atan2(cx*sz+sx*sy*cz, cy*cz)

    return (erx, ery, erz)

def __cross_mx(r) :
    return np.array(
        [[ 0   , -r[2],  r[1] ],
         [ r[2],   0  , -r[0] ],
         [-r[1],  r[0],    0  ]] )

def rotoTranslateInertia(inertia, tr, R) :
    mass = inertia['mass']
    com  = inertia['com']
    vec  = com - tr

    com_x = __cross_mx(com)
    vec_x = __cross_mx(vec)

    ixx = inertia['Ix']
    iyy = inertia['Iy']
    izz = inertia['Iz']
    ixy = inertia['Ixy']
    ixz = inertia['Ixz']
    iyz = inertia['Iyz']
    tensor = np.array( [[ ixx, -ixy, -ixz],
                        [-ixy,  iyy, -iyz],
                        [-ixy, -iyz,  izz] ])
    tensor = tensor - mass * (com_x @ com_x.T - vec_x @ vec_x.T)

    tensor2 = R @ tensor @ R.T
    com2 = R @ vec
    ret = {}
    ret['mass'] = mass
    ret['com'] = com2
    ret['Ix']  =  tensor2[0,0]
    ret['Iy']  =  tensor2[1,1]
    ret['Iz']  =  tensor2[2,2]
    ret['Ixy'] = -tensor2[0,1]
    ret['Ixz'] = -tensor2[0,2]
    ret['Iyz'] = -tensor2[1,2]
    return ret


'''
Simply reads the XML file and stores the links/joints data, no conversions
'''
class URDFWrapper :
    class Link:
        def __init__(self, name):
            self.name    = name
            self.inertia = None
            self.parent  = None
            self.supportingJoint = None
    class Joint:
        def __init__(self, name):
            self.name = name
            self.type  = None
            self.frame = None
            self.parent= None
            self.child = None
            self.predec_H_joint = np.identity(4)

    iMomentsLabels = ['ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz']

    def __init__(self, urdfInFile):
        root = ET.parse(urdfInFile)
        self.robotName = root.getroot().get('name')

        linkNodes  = root.findall("link")
        jointNodes = root.findall("joint")

        self.links  = ODict()
        self.joints = ODict()
        self.frames = ODict()

        for nodelink in linkNodes:
            name = nodelink.get('name')
            link = URDFWrapper.Link( name )
            link.inertia = self.readInertialData(nodelink)
            self.links[name] = link

        for nodejoint in jointNodes:
            name = nodejoint.get('name')
            joint = URDFWrapper.Joint( name )
            joint.type  = nodejoint.get('type')
            joint.frame = self.readJointFrameData( nodejoint )
            joint.predec_H_joint[:3,:3] = getR_extrinsicXYZ( * joint.frame['rpy'] )
            joint.predec_H_joint[:3,3]  = np.array( joint.frame['xyz'] )
            joint.parent= nodejoint.find('parent').get('link')
            joint.child = nodejoint.find('child').get('link')

            # Note I keep URDF nomenclature ("parent" and "child") just to
            # stress the bond with the source URDF XML file. I will later use
            # the more appropriate terms (e.g. "predecessor")

            self.joints[name] = joint

            predecessor = self.links[ joint.parent ]
            successor   = self.links[ joint.child ]
            successor.parent = predecessor # a Link instance, not a name
            successor.supportingJoint = joint

    def readInertialData(self, linkNode):
        params = dict()
        paramsNode = linkNode.find('inertial')

        # Default inertia parameters if the URDF does not have the data
        if paramsNode == None :
            params['mass'] = 0.0
            params['xyz']  = (0.0, 0.0, 0.0)
            for m in URDFWrapper.iMomentsLabels :
                params[m] = 0.0
            return params

        mass = float(paramsNode.find('mass').get('value'))

        xyz = (0.0, 0.0, 0.0)
        originNode = paramsNode.find('origin')
        if originNode != None :
            comstr = originNode.get('xyz')
            if(comstr != None) :
                xyz = tuple([float(x) for x in comstr.split()])

            # We cannot deal with non-zero values for the 'rpy' attribute
            rpystr = originNode.get('rpy')
            if(rpystr != None) :
                tmp = [float(x) for x in rpystr.split()]
                if(sum(tmp) != 0) :
                    logger.warning('The rpy attribute in the inertial section is not yet supported (link ' + linkNode.get('name') + '). Ignoring it.')

        moments = paramsNode.find('inertia')
        for m in URDFWrapper.iMomentsLabels :
            params[m] = float(moments.get(m))

        params['mass'] = mass
        params['xyz']  = xyz
        return params


    def readJointFrameData(self, jointNode):
        params = dict()

        # URDF defaults:
        params['xyz'] = (0,0,0)
        params['rpy'] = (0,0,0)

        frameNode = jointNode.find('origin')
        if frameNode != None :
            xyz_node = frameNode.get('xyz')
            if xyz_node != None :
                params['xyz'] = tuple([float(x) for x in xyz_node.split()])
            rpy_node = frameNode.get('rpy')
            if rpy_node != None :
                params['rpy'] = tuple([float(x) for x in rpy_node.split()])

        axis_node = jointNode.find('axis')
        if axis_node != None :
            params['axis'] = tuple([float(x) for x in axis_node.get('xyz').split()])
        else :
            params['axis'] = (1,0,0) # URDF default

        return params



class Converter :
    '''Reads the model from a URDFWrapper instance, and applies the necessary conversions
    '''
    class Frame :
        def __init__(self):
            self.H   = np.identity(4)    # Homogeneous coordinate transform, from this-instance-coordinates to some link coordinates
            self.rot = (0.0, 0.0, 0.0)   # Intrinsic rx, ry, rz angles
            self.tr  = self.H[0:3,3]     # View of the translation vector

    class Link :
        def __init__(self, namestr):
            self.name    = namestr
            self.parent  = None
            self.parentJ = None
            self.children= list()
            self.inertia = dict()
            self.frames  = dict()
            self.rcg_R_urdf = np.identity(3)

    class Joint :
        def __init__(self, namestr):
            self.name = namestr
            self.type = 'revolute'
            self.predecessor = None
            self.successor  = None
            self.frame = Converter.Frame()


    @staticmethod
    def toValidID( name ) :
        return name.replace('-', '__')

    def __init__(self, urdf, options) :
        self.robotName = urdf.robotName
        self.links  = ODict()
        self.joints = ODict()
        self.frames = ODict()

        for urdfname in urdf.links.keys() :
            name = self.toValidID( urdfname )
            link = Converter.Link( name )
            self.links[name] = link

        for jname in urdf.joints.keys() :
            urdfjoint = urdf.joints[jname]
            name = self.toValidID( jname )
            joint= Converter.Joint( name )
            joint.type = urdfjoint.type

            joint.predecessor = self.links[ self.toValidID(urdfjoint.parent) ]
            joint.successor   = self.links[ self.toValidID(urdfjoint.child)  ]
            self.convertJointFrame(joint, urdfjoint)

            joint.successor.parent = joint.predecessor
            joint.successor.parentJ= joint
            joint.predecessor.children.append( (joint.successor, joint) )
            self.joints[name] = joint

        orphans = [l for l in self.links.values() if l.parent==None]
        if len(orphans)==0 :
            logger.fatal("Could not find any root link (i.e. a link without parent).")
            logger.fatal("Check for kinematic loops.")
            print("Error, no root link found. Aborting", file=sys.stderr)
            sys.exit(-1)
        if len(orphans) > 1 :
            logger.warning("Found {0} links without parent, only one expected".format(len(orphans)))
            logger.warning("Any robot model must have exactly one root element.")
            logger.warning("This might lead to unexpected results.")
        self.root = orphans[0]

        self.leafs = [l for l in self.links.values() if len(l.children)==0]

        # Conversion of the inertia must happen after the joint frame conversion,
        # which determines the required coordinate transforms
        for urdfname in urdf.links.keys() :
            name = self.toValidID( urdfname )
            self.convertInertialData(self.links[name], urdf.links[urdfname].inertia)

        if options.prunefixed :
            # Let's first explicitly check for the nasty case where the root is
            # a dummy link, connected via a fixed joint to the first link
            self._collapseDummyRoots()

            self._pruneDummies(options)

    def _collapseDummyRoots(self):
        root = self.root
        keepGoing = True
        while self.isDummyLink(root) and keepGoing :
            if len(root.children) > 1 :
                logger.warning("Detected dummy root link ({0}) with multiple children; cannot collapse".format(root.name))
                keepGoing = False
                for childPair in root.children :
                    childPair[1].__preserve = True
            else :
                childSpec = root.children[0]
                joint = childSpec[1]
                child = childSpec[0]
                if joint.type != 'fixed' :
                    logger.warning("Detected dummy root link ({0}) supporting a non-fixed joint ({1})".format(root.name, joint.name))
                    keepGoing = False
                    joint.__preserve = True
                else :
                    # We have a dummy root link, with only one child connected via fixed joint.
                    # Let's delete it and replace the root
                    logger.info("Deleting dummy pair '{0}'-'{1}', root replaced with '{2}'".format(
                        root.name, joint.name, child.name))
                    root = child
                    del self.links[root.name]
                    del self.joints[joint.name]

        self.root = root

    def _pruneDummies(self, options):
        for leaf in self.leafs :
            link = leaf

            while True :
                joint  = link.parentJ
                parent = link.parent

                if joint is None or parent is None:
                    break
                if joint.type != 'fixed' :
                    break

                logger.debug("Trying to collapse link '{0}', connected by joint '{1}'".format(link.name, joint.name))
                if options.toframes :
                    # We need to "move" the frames associated with 'link' into
                    # the parent frames. First of all, the implicit link frame,
                    # which is the same as the supporting-joint frame:
                    parent.frames[link.name] = joint.frame

                    # Then the additional custom frames on the link; for these
                    # ones we must perform a coordinate transform
                    for ufr in link.frames.keys() :
                        original = link.frames[ufr]
                        shiftedup= Converter.Frame()
                        # We need the [:,:] to assign values to the same memory
                        # location, because the translation attribute is a view
                        # of H. If we change H, the view will be inconsistent
                        shiftedup.H[:,:] = np.matmul( joint.frame.H, original.H )
                        shiftedup.rot = getIntrinsicXYZFromR( shiftedup.H[0:3,0:3] )
                        parent.frames[ufr] = shiftedup

                if options.lumpinertia :
                    # Keep in mind that at this point all the inertia properties
                    # are in robcogen format, that is, in link coordinates. And
                    # the link frame is the same as the supporting-joint frame
                    if link.inertia['mass'] != 0 :
                        loadMe = parent.inertia
                        parent_R_link = getR_intrinsicXYZ( *joint.frame.rot )

                        # The translation we need is the position of the parent
                        # link frame relative to the joint frame, in joint frame
                        # coordinates
                        tr = - parent_R_link.T @ joint.frame.tr
                        # Transform the inertia of the link in the coordinate
                        # system of the parent link
                        addMe = rotoTranslateInertia(link.inertia, tr, parent_R_link)
                        m1 = loadMe['mass']
                        m2 = addMe['mass']

                        loadMe['mass'] = m1 + m2
                        loadMe['Ix']  = loadMe['Ix']  + addMe['Ix']
                        loadMe['Iy']  = loadMe['Iy']  + addMe['Iy']
                        loadMe['Iz']  = loadMe['Iz']  + addMe['Iz']
                        loadMe['Ixy'] = loadMe['Ixy'] + addMe['Ixy']
                        loadMe['Ixz'] = loadMe['Ixz'] + addMe['Ixz']
                        loadMe['Iyz'] = loadMe['Iyz'] + addMe['Iyz']
                        loadMe['com'] = (loadMe['com']*m1 + addMe['com']*m2)/(m1+m2)

                # The actual clean up of the traces of 'link'
                del self.links[link.name]
                del self.joints[joint.name]
                parent.children.remove( (link, joint) )
                link = parent # recursively keep going up

        # Reconstruct the leafs array, after the pruning
        self.leafs = [l for l in self.links.values() if len(l.children)==0]


    def isDummyLink(self, link):
        immaterial = link.inertia['mass'] == 0.0
        fixedj = False
        if link.parentJ is not None :
            fixedj = (link.parentJ.type == 'fixed')

        return (immaterial and fixedj)

    def convertInertialData(self, link, urdfParams):
        iin = {}
        iin['mass'] = urdfParams['mass']
        iin['com']  = np.zeros(3)
        iin['Ix']   =  urdfParams['ixx']
        iin['Iy']   =  urdfParams['iyy']
        iin['Iz']   =  urdfParams['izz']
        iin['Ixy']  = -urdfParams['ixy']
        iin['Ixz']  = -urdfParams['ixz']
        iin['Iyz']  = -urdfParams['iyz']

        tr = -np.array(urdfParams['xyz'])
        iout = rotoTranslateInertia(iin, tr, link.rcg_R_urdf)
        link.inertia = iout

    def convertJointFrame(self, joint, urdfjoint):
        rpy = urdfjoint.frame['rpy']

        # Rotation matrix from URDF joint frame to URDF link frame
        urdflink_X_urdfjoint = getR_extrinsicXYZ(* rpy )

        # Rotation matrix from URDF joint frame to RobCoGen link frame
        rcglink_X_urdfjoint = np.matmul(joint.predecessor.rcg_R_urdf, urdflink_X_urdfjoint)

        '''
        If the URDF joint is fixed, there is no axis. In that case we only need
        to get the intrinsic rotation parameters (robcogen), from the complete
        rotation matrix we already computed above.
        Otherwise, we need to get the coordinates of the joint axis and make
        sure to rotate the robcogen frame so as to align its Z axis with the
        joint axis.
        '''
        if urdfjoint.type == "fixed" :
            (rx, ry, rz) = getIntrinsicXYZFromR( rcglink_X_urdfjoint )
        else :
            axis = np.array( urdfjoint.frame['axis'] )

            # The joint axis in robcogen-link-frame coordinates
            axis_linkframe = np.matmul(rcglink_X_urdfjoint, axis)

            '''
            We want to find the intrinsic rotations rx ry rz for the joint frame in the
            RobCoGen model. These rotations must be such that the Z axis of the
            resulting frame is aligned with the joint axis, computed above. This
            constraint and the expression of the full matrix (see method above), give us
            the equations for rx ry :

                 sin(ry)         = axis_x
               - sin(rx) cos(ry) = axis_y
                 cos(rx) cos(ry) = axis_z
            '''

            ry = math.asin( axis_linkframe[0] )
            cy = math.cos(ry)
            if round(cy,5) != 0.0 :
                rx = math.asin( - axis_linkframe[1] / cy )
            else:
                rx = 0.0
            rz = 0.0;
            if round(math.cos(rx)*math.cos(ry),5) != round(axis_linkframe[2],5) :
                logger.warning("possible inconsistency in the joint frame rotation")

        # Rotation matrix from RobCoGen joint frame to link frame
        rcglink_X_rcgjoint = getR_intrinsicXYZ(rx, ry, rz)

        # Rotation matrix from URDF joint frame to RobCoGen joint frame.
        # We can interpret this one as the rotation difference between the joint
        #  frames in the two models.
        R = np.matmul(rcglink_X_rcgjoint.transpose() , rcglink_X_urdfjoint)

        # The best we can do, at this point, is to check whether the difference
        # between the two frames is simply a rotation about z (for a revolute
        # joint), and, if so, add it to the parameters (affecting the zero
        # configuration only). We do this to "minimize" the variation with
        # respect to the URDF joint frame.
        if joint.type == 'revolute' :
            roundDigits = 5
            Z = np.array([0,0,1])
            Rz = R[:,2]

            # If we have a pure rotation about Z ...
            if np.equal( np.round(Rz, roundDigits), Z).all() :
                # Special case angle=PI, for which the formulas below do not work
                if round(R[0,0],roundDigits) == -1 and round(R[1,1],roundDigits) == -1 :
                    rz = math.pi
                else:
                    diffaxis = np.array( [ R[2,1] - R[1,2],  R[0,2] - R[2,0], R[1,0] - R[0,1] ] )
                    norm     = np.linalg.norm( diffaxis )
                    if norm > 1e-5 :
                        rz = math.atan2( norm, R.trace()-1 )

                        axisnorm = np.round( diffaxis/norm, 5 ) # normalized axis, rounded
                        # Check the inversion of the axis due to negative rotation angles
                        # Our axis must be Z, not -Z
                        if axisnorm[2] < 0 :
                            axisnorm[2] = -axisnorm[2]
                            rz = -rz

                        if not np.equal( axisnorm, Z).all() :
                            # We checked above it's a pure rotation about Z, so that
                            # should be confirmed here too...
                            msg = "Possible inconsistency in the " \
                                + joint.name + " joint frame rotation"
                            logger.warning(msg)

                # Now that we have possibly changed rz, recompute the joint transform
                # and the rotation difference with the URDF transform
                rcglink_X_rcgjoint = getR_intrinsicXYZ(rx, ry, rz)
                R = np.matmul(rcglink_X_rcgjoint.transpose(), rcglink_X_urdfjoint)

        # Save the transformation from joint coordinates to link coordinates
        joint.frame.tr[:] = np.matmul(joint.predecessor.rcg_R_urdf, urdfjoint.frame['xyz'])
        joint.frame.rot   = (rx, ry, rz)
        joint.frame.H[0:3,0:3] = rcglink_X_rcgjoint

        # Save the rotation difference between robcogen and urdf link frames
        joint.successor.rcg_R_urdf = R


'''
Rounds the floating point numbers for pretty printing
'''
class NumFormatter :
    def __init__(self, round_digits=6, pi_round_digits=5):
        self.round_decimals = round_digits
        self.pi_round_decimals = pi_round_digits

        self.roundedPI     = round(math.pi,   self.pi_round_decimals)
        self.roundedHalfPI = round(math.pi/2, self.pi_round_decimals)

        self.formatStr = '0:.' + str(self.round_decimals)

    def float2str(self, num, angle=False ) :
        if angle :
            value = round(num, self.pi_round_decimals)
            sign  = "-" if value<0 else ""
            if abs(value) == self.roundedPI :
                return sign+"PI"
            if abs(value) == self.roundedHalfPI :
                return sign+"PI/2.0"

        num = round(num, self.round_decimals)
        num += 0  # this trick avoids the annoying '-0.0' (minus zero)

        # I can't use the '.<n>f' right away because it inserts trailing zeros
        # to fill up <n> decimal positions, which is annoying.
        # So, I first apply standard formatting (no trailing zeros), and then
        # get rid of the scientific notation in case it has been used.
        ret = ( "{" + self.formatStr + "}" ).format( num )
        if "e" in ret:
            ret = ( "{" + self.formatStr + "f}" ).format( num )
        return ret


'''
Writes the Kinematics-DSL document corresponding to the given Converter instance
'''
class Serializer :

    def __init__(self, outfile, numFormatter=NumFormatter() ):
        self.__ind = 0
        self.linkID = 1
        self.file = outfile
        self.formatter = numFormatter

    def vec3Str(self, prefix, tupl, angles=False):
        return prefix + '({0[0]:s}, {0[1]:s}, {0[2]:s})'.format(
            [self.formatter.float2str(s, angle=angles) for s in tupl] )

    def indent(self):
        self.__ind += 4
    def indentback(self):
        self.__ind -= 4
    def myprint(self, text):
        print(self.__ind*' ', text, sep='', file=self.file)

    def _printFrame(self, tr, rot) :
        self.myprint( self.vec3Str('translation = ', tr ) )
        self.myprint( self.vec3Str('rotation    = ', rot, angles=True) )

    def _blockStart(self, name):
        self.myprint(name + ' {')
        self.indent()

    def _blockEnd(self):
        self.indentback()
        self.myprint('}')

    def printJoint(self, j):
        if(j.type == 'prismatic') :
            keyw = 'p_joint'
        else :
            keyw = 'r_joint'
        self._blockStart(keyw + ' ' + j.name)
        self._blockStart('ref_frame')
        self._printFrame( j.frame.tr, j.frame.rot )
        self._blockEnd()
        self._blockEnd()

    def printInertiaParams(self, params):
        self._blockStart('inertia_properties')
        self.myprint('mass = ' + self.formatter.float2str(params['mass']) )
        self.myprint( self.vec3Str('CoM = ', params['com']) )
        for m in ['Ix', 'Iy', 'Iz', 'Ixy', 'Ixz', 'Iyz'] :
            self.myprint(m + (3-len(m))*' ' + '= ' + self.formatter.float2str(params[m]) )
        self._blockEnd()

    def printChildren(self, link):
        self._blockStart('children')
        for child in link.children :
            self.myprint( child[0].name + ' via ' + child[1].name )
        self._blockEnd()

    def printUserFrames(self, link):
        urdfRots  = getIntrinsicXYZFromR(link.rcg_R_urdf)
        urdfFrame =  any( [math.fabs(x)>1e-5 for x in urdfRots] )
        if urdfFrame or (len(link.frames)>0) :
            self._blockStart('frames')
            for uf in link.frames.keys() :
                self._blockStart(uf)
                self._printFrame(link.frames[uf].tr, link.frames[uf].rot)
                self._blockEnd()

            if urdfFrame :
                self._blockStart('urdf_' + link.name)
                self._printFrame( (0.0,0.0,0.0), urdfRots )
                self._blockEnd()
            self._blockEnd()

    def printLinks_DFS(self, root ) : #DFS = Depth-First-Search
        for child in root.children:
            link = child[0]
            self._blockStart('link ' + link.name)
            self.myprint('id = ' + self.linkID.__str__())
            self.printInertiaParams(link.inertia)
            self.printChildren(link)
            self.printUserFrames(link)
            self._blockEnd()
            self.myprint('\n')

            self.linkID += 1
            self.printLinks_DFS(link)


    def writeModel(self, converted):
        self.myprint('Robot ' + converted.robotName + '\n{\n')
        robotBase = converted.root
        self._blockStart('RobotBase ' + robotBase.name)
        self.printInertiaParams(robotBase.inertia)
        self.printChildren(robotBase)
        self.printUserFrames(robotBase)
        self._blockEnd()
        self.myprint('\n')

        self.printLinks_DFS(robotBase)

        for j in converted.joints.values():
            self.printJoint(j)
            self.myprint('')
        self.myprint('}\n')


def urdfdbg_linkOrigin(urdf, eelinkname):
    logger.debug("Entering urdfdbg_linkOrigin() function ...")
    H = np.identity(4)

    currentLink  = urdf.links[eelinkname]
    while currentLink is not None :
        currentJoint = currentLink.supportingJoint
        logger.debug("Link : " + currentLink.name)
        if currentJoint is not None :
            logger.debug("Joint: " + currentJoint.name)
            H = np.matmul( currentJoint.predec_H_joint , H )
        currentLink  = currentLink.parent

    print( np.round(H[:3,3], 5) )


logLevels = {}
logLevels['debug']   = logging.DEBUG
logLevels['info']    = logging.INFO
logLevels['warning'] = logging.WARNING
logLevels['error']   = logging.ERROR


if __name__ == "__main__" :
    argparser = argparse.ArgumentParser(
        description='Convert a URDF model to a Kinematics-DSL model')

    argparser.add_argument('urdf', metavar='URDF-input',
            help='path of the URDF input file')
    argparser.add_argument('-o', '--output',
            help='destination file (defaults to stdout)')
    argparser.add_argument('--digits',
            type=int,
            help='max number of digits for the fractional part of a real number (default 6)',
            default=6)
    argparser.add_argument('--pi-digits',
            type=int,
            help='number of digits of the fractional part of an angle used to determine if it is equal to PI (default 5)',
            default=5)
    argparser.add_argument('--prune-fixed-joints', dest='prunefixed',
            action='store_true',
            help='prune fixed joints and child links - see also the following options')

    argparser.add_argument('--to-frames'   , dest='toframes', action='store_true')
    argparser.add_argument('--no-to-frames', dest='toframes', action='store_false',
            help='convert pruned links to custom frames in the parent link; defaults to true')

    argparser.add_argument('--lump-inertia', dest='lumpinertia', action='store_true')
    argparser.add_argument('--no-lump-inertia', dest='lumpinertia',
            action='store_false',
            help='''propagate up the tree the inertia of pruned links; defaults to true''')
    argparser.set_defaults(prunefixed=False)
    argparser.set_defaults(lumpinertia=True)
    argparser.set_defaults(toframes=True)

    argparser.add_argument('--log-level', type=str, dest='loglevel',
            default='warning',
            help='logging level, chosen among debug, info, warning, error (defaults to warning)')
    group = argparser.add_argument_group('URDF inspection', 'Misc information about the given URDF (no conversion performed)')
    group.add_argument('--link-origin', metavar="LINK",
            type=str,
            help='print the origin of the frame of LINK in base coordinates, for the zero configuration'
            )
    args = argparser.parse_args()

    logging.basicConfig(level= logLevels[args.loglevel])
    logger = logging.getLogger(__name__)

    ofile = sys.stdout
    if( args.output is not None) :
        ofile = open(args.output, 'w')

    urdf = URDFWrapper( args.urdf )
    if args.link_origin is not None :
        urdfdbg_linkOrigin(urdf, args.link_origin)
    else :
        conv = Converter( urdf, args )
        form = NumFormatter( round_digits=args.digits, pi_round_digits=args.pi_digits)
        ser = Serializer(ofile, numFormatter=form)
        ser.writeModel(conv)


