#!/usr/bin/env python
'''
Created on August 2018

@author: Marco Frigerio
'''
from __future__ import print_function
import logging, sys, argparse, math
import xml.etree.ElementTree as ET
from collections import OrderedDict as ODict
import numpy as np



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
        frameNode = jointNode.find('origin')

        params = dict()
        params['xyz'] = tuple([float(x) for x in frameNode.get('xyz').split()])
        params['rpy'] = tuple([float(x) for x in frameNode.get('rpy').split()])

        axis_node = jointNode.find('axis')
        if axis_node != None :
            params['axis'] = tuple([float(x) for x in axis_node.get('xyz').split()])
        else :
            params['axis'] = (1,0,0) # URDF default

        return params



'''
Reads the model from a URDFWrapper instance, and applies the necessary conversions
'''
class Converter :
    class Link :
        def __init__(self, namestr):
            self.name    = namestr
            self.parent  = None
            self.children= list()
            self.inertia = dict()
            self.rcg_R_urdf = np.identity(3)

    class Joint :
        def __init__(self, namestr):
            self.name = namestr
            self.type = 'revolute'
            self.predecessor = None
            self.successor  = None
            self.frame = dict()

    @staticmethod
    def toValidID( name ) :
        return name.replace('-', '__')

    def __init__(self, urdf) :
        self.robotName = urdf.robotName
        self.links  = ODict()
        self.joints = ODict()

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
            joint.predecessor.children.append( (joint.successor, joint) )
            self.joints[name] = joint

        # Conversion of the inertia must happen after the joint frame conversion,
        # which determines the required coordinate transforms
        for urdfname in urdf.links.keys() :
            name = self.toValidID( urdfname )
            self.convertInertialData(self.links[name], urdf.links[urdfname].inertia)


    def convertInertialData(self, link, urdfParams):
        com = urdfParams['xyz']
        mass= urdfParams['mass']

        # Parallel axis theorem: transforms the inertia moments expressed in the
        # COM frame into the link frame. Note the sign swap for the centrifugal
        # moments, as the URDF stores the element of the tensor, not the moment
        # itself.
        comx = com[0]
        comy = com[1]
        comz = com[2]
        ixx =  urdfParams['ixx'] + mass * (comy*comy + comz*comz)
        iyy =  urdfParams['iyy'] + mass * (comx*comx + comz*comz)
        izz =  urdfParams['izz'] + mass * (comx*comx + comy*comy)
        ixy = -urdfParams['ixy'] + mass * comx * comy
        ixz = -urdfParams['ixz'] + mass * comx * comz
        iyz = -urdfParams['iyz'] + mass * comy * comz

        tensor = np.array( [[ ixx, -ixy, -ixz],
                            [-ixy,  iyy, -iyz],
                            [-ixy, -iyz,  izz] ])
        tensor_rcg = np.matmul( link.rcg_R_urdf, np.matmul( tensor, link.rcg_R_urdf.transpose()) )

        link.inertia['Ix']  =  tensor_rcg[0,0]
        link.inertia['Iy']  =  tensor_rcg[1,1]
        link.inertia['Iz']  =  tensor_rcg[2,2]
        link.inertia['Ixy'] = -tensor_rcg[0,1]
        link.inertia['Ixz'] = -tensor_rcg[0,2]
        link.inertia['Iyz'] = -tensor_rcg[1,2]

        link.inertia['mass']= mass
        link.inertia['com'] = np.matmul(link.rcg_R_urdf, com)


    def convertJointFrame(self, joint, urdfjoint):
        rpy = urdfjoint.frame['rpy']
        sx = math.sin(rpy[0])
        cx = math.cos(rpy[0])
        sy = math.sin(rpy[1])
        cy = math.cos(rpy[1])
        sz = math.sin(rpy[2])
        cz = math.cos(rpy[2])
        '''
        This is the rotation matrix base_R_rotated, where 'rotated' is obtained from
        'base' with the **extrinsic** rotations rx, ry, and rz

        cos(ry) cos(rz)    sin(rx) sin(ry) cos(rz) - cos(rx) sin(rz)    sin(rx) sin(rz) + cos(rx) sin(ry) cos(rz)
        cos(ry) sin(rz)    sin(rx) sin(ry) sin(rz) + cos(rx) cos(rz)    cos(rx) sin(ry) sin(rz) - sin(rx) cos(rz)
           - sin(ry)                    sin(rx) cos(ry)                              cos(rx) cos(ry)
        '''
        urdflink_X_urdfjoint = np.array(
            [[cy*cz,  cz*sx*sy - cx*sz,  sx*sz + cx*cz*sy],
             [cy*sz,  sx*sy*sz + cx*cz,  cx*sy*sz - cz*sx],
             [ - sy,        cy*sx     ,        cx*cy      ]] )

        rcglink_X_urdfjoint = np.matmul(joint.predecessor.rcg_R_urdf, urdflink_X_urdfjoint)

        '''
        If the URDF joint is fixed, there is no axis. In that case we just
        convert the extrinsic rotation parameters (URDF) to intrinsic rotations
        (robcogen), to have the same frame.
        Otherwise, we need to get the coordinates of the joint axis and make
        sure to rotate the robcogen frame so as to align its Z axis with the
        joint axis.
        '''
        if urdfjoint.type == "fixed" :
            (rx, ry, rz) = _intrinsic2extrinsic_XYZ( * rpy )
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
                logging.warning("possible inconsistency in the joint frame rotation")

        rcglink_X_rcgjoint = getR_intrinsicXYZ(rx, ry, rz)
        R = np.matmul(rcglink_X_rcgjoint.transpose() , rcglink_X_urdfjoint)

        # The best we can do, at this point, is to check whether the difference
        # between the two frames is simply a rotation about z, because (for a revolute
        # joint), we can always add a rotation about z without changing the rest
        # (the zero configuration gets affected)
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
                        axisnorm = np.round( diffaxis/norm, 5 ) # normalized axis, rounded
                        if not np.equal( axisnorm, Z).all() :
                            # We checked above it's a pure rotation about Z, so that
                            # should be confirmed here too...
                            logging.warning("possible inconsistency in the joint frame rotation")
                        rz = math.atan2( norm, R.trace()-1 )

                # Now that we have possibly changed rz, recompute the joint transform
                # and the rotation difference with the URDF transform
                rcglink_X_rcgjoint = getR_intrinsicXYZ(rx, ry, rz)
                R = np.matmul(rcglink_X_rcgjoint.transpose(), rcglink_X_urdfjoint)

        joint.frame['translation'] = np.matmul(joint.predecessor.rcg_R_urdf, urdfjoint.frame['xyz'])
        joint.frame['rotation'] = (rx, ry, rz)
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
        self.myprint( self.vec3Str('translation= ', tr ) )
        self.myprint( self.vec3Str('rotation   = ', rot, angles=True) )

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
        self._printFrame( j.frame['translation'], j.frame['rotation'] )
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

    def printLinks_DFS(self, root ) : #DFS = Depth-First-Search
        for child in root.children:
            link = child[0]
            self._blockStart('link ' + link.name)
            self.myprint('id = ' + self.linkID.__str__())
            self.printInertiaParams(link.inertia)
            self.printChildren(link)

            rots = getIntrinsicXYZFromR(link.rcg_R_urdf)
            if math.fsum( [math.fabs(x) for x in rots] ) > 0 :
                self._blockStart('frames')
                self._blockStart('urdf_' + link.name)
                self._printFrame( (0.0,0.0,0.0), rots )
                self._blockEnd()
                self._blockEnd()

            self._blockEnd()
            self.myprint('\n')

            self.linkID += 1
            self.printLinks_DFS(link)


    def writeModel(self, converted):
        self.myprint('Robot ' + converted.robotName + '\n{\n')
        robotBase = [l for l in converted.links.values() if l.parent == None][0]
        self._blockStart('RobotBase ' + robotBase.name)
        self.printInertiaParams(robotBase.inertia)
        self.printChildren(robotBase)
        self._blockEnd()
        self.myprint('\n')

        self.printLinks_DFS(robotBase)

        for j in converted.joints.values():
            self.printJoint(j)
            self.myprint('')
        self.myprint('}\n')



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

    args = argparser.parse_args()

    file = sys.stdout
    if( args.output is not None) :
        file = open(args.output, 'w')

    urdf = URDFWrapper( args.urdf )
    conv = Converter( urdf )
    form = NumFormatter( round_digits=args.digits, pi_round_digits=args.pi_digits)
    ser = Serializer(file, numFormatter=form)
    ser.writeModel(conv)


def debug(urdf, eelinkname):
    logging.debug("Entering urdf2kindsl debug function ...")
    H = np.identity(4)

    currentLink  = urdf.links[eelinkname]
    while currentLink is not None :
        currentJoint = currentLink.supportingJoint
        logging.debug("Link : " + currentLink.name)
        if currentJoint is not None :
            logging.debug("Joint: " + currentJoint.name)
            H = np.matmul( currentJoint.predec_H_joint , H )
        currentLink  = currentLink.parent

    print( np.round(H, 5) )

