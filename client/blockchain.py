from typing import List
import hashlib


class Node:
    def __init__(self, left, right, value: str, content, is_copied=False) -> None:
        self.left: Node = left
        self.right: Node = right
        self.value = value
        self.content = content
        self.is_copied = is_copied

    @staticmethod
    def hash(val: str) -> str:
        return hashlib.sha256(val.encode('utf-8')).hexdigest()

    def __str__(self):
        return (str(self.value))

    def copy(self):
        return Node(self.left, self.right, self.value, self.content, True)


class MerkleTree:
    def __init__(self, values: List[str]) -> None:
        self.__buildTree(values)

    def __buildTree(self, values: List[str]) -> None:

        leaves: List[Node] = [Node(None, None, Node.hash(e), e) for e in values]
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1].copy())  # duplicate last elem if odd number of elements
        self.root: Node = self.__buildTreeRec(leaves)

    def __buildTreeRec(self, nodes: List[Node]) -> Node:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1].copy())  # duplicate last elem if odd number of elements
        half: int = len(nodes) // 2

        if len(nodes) == 2:
            return Node(nodes[0], nodes[1], Node.hash(nodes[0].value + nodes[1].value),
                        nodes[0].content + "+" + nodes[1].content)

        left: Node = self.__buildTreeRec(nodes[:half])
        right: Node = self.__buildTreeRec(nodes[half:])
        value: str = Node.hash(left.value + right.value)
        content: str = f'{left.content}+{right.content}'
        return Node(left, right, value, content)

    def printTree(self) -> None:
        self.__printTreeRec(self.root)

    def __printTreeRec(self, node: Node) -> None:
        if node != None:
            if node.left != None:
                print("Left: " + str(node.left))
                print("Right: " + str(node.right))
            else:
                print("Input")

            if node.is_copied:
                print('(Padding)')
            print("Value: " + str(node.value))
            print("Content: " + str(node.content))
            print("")
            self.__printTreeRec(node.left)
            self.__printTreeRec(node.right)

    def getRootHash(self) -> str:
        return self.root.value


class Block:
    def __init__(self, prevhashVal, data, merkleRoot=0, hashVal=""):
        self.hashVal = hashVal
        self.merkleroot = merkleRoot
        self.prevHashVal = prevhashVal
        self.data = data

    def calculateHash(self):
        hashObj = hashlib.sha256(self.data.encode())
        return hashObj.hexdigest()

    def retHashVal(self):
        return self.hashVal


class Blockchain:
    prevHash = "420"

    def __init__(self):
        self.chain = []
        hashVal = hashlib.sha256("genesis block".encode()).hexdigest()
        currBlock = Block(Blockchain.prevHash, "genesis block", 6464, hashVal)
        Blockchain.prevHash = hashVal
        self.chain.append(currBlock)

    def addBlock(self, data):

        currBlock = Block(Blockchain.prevHash, data)
        hashVal = currBlock.calculateHash()
        currBlock.hashVal = hashVal
        currBlock.merkleroot = MerkleTree(self.generateListOfHashes()).getRootHash()

        # print("Appending block")
        self.chain.append(currBlock)

        Blockchain.prevHash = hashVal

    def generateMerkleRoot(self):
        pass

    def verifyChain(self) -> bool:
        chainLength = len(self.chain)
        for i in range(1, chainLength):
            currBlock = self.chain[i]
            prevBlock = self.chain[i]
            if currBlock.hashVal != currBlock.calculateHash():
                return False
            if prevBlock.hashVal != currBlock.prevHashVal:
                return False

        return True

    def indexBlock(self, pos) -> Block:
        if 2 > pos > len(self.chain):
            return None
        return self.chain[pos - 1]

    def displayChain(self):

        chain = self.chain
        for i in range(len(chain)):
            print("Previous Hash Value:", chain[i].prevHashVal)
            print("Merkle Root:", chain[i].merkleroot)
            print("Current Hash Value:", chain[i].hashVal)
            print("Data:", chain[i].data)
            print("\n")
    def generateListOfHashes(self):
        hashList = []
        for i in self.chain :
            hashList.append(i.retHashVal())
        return hashList

#test
def mixmerkletree() -> None:
    elems = ["GeeksforGeeks", "A", "Computer", "Science", "Portal", "For", "Geeks"]
    # as there are odd number of inputs, the last input is repeated
    print("Inputs: ")
    print(*elems, sep=" | ")
    print("")
    mtree = MerkleTree(elems)
    print("Root Hash: " + mtree.getRootHash() + "\n")
    # mtree.printTree()


# myChain = Blockchain()

# myChain.addBlock("hello")
# myChain.addBlock("this is me")
# myChain.addBlock("hello world")
#
# myChain.displayChain()
# mixmerkletree()
# hashList = []
# hashList.append(myChain.generateListOfHashes())
# print(myChain.generateListOfHashes())
#
# mtree = MerkleTree(myChain.generateListOfHashes())
#
# print(mtree.getRootHash())
