---
title : '刷题日记'
date : '2025-03-14T17:49:08+08:00'
draft : false
description: 栈与队列
categories:
    - 上机
    - C/C++
tags:
    - 代码随想录
---

#### 1.26-1 [用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/)

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（`push`、`pop`、`peek`、`empty`）：

实现 `MyQueue` 类：

- `void push(int x)` 将元素 x 推到队列的末尾
- `int pop()` 从队列的开头移除并返回元素
- `int peek()` 返回队列开头的元素
- `boolean empty()` 如果队列为空，返回 `true` ；否则，返回 `false`

**说明：**

- 你 **只能** 使用标准的栈操作 —— 也就是只有 `push to top`, `peek/pop from top`, `size`, 和 `is empty` 操作是合法的。
- 你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。

**示例 1：**

```
输入：
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
输出：
[null, null, null, 1, 1, false]

解释：
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); //return false
```

**提示：**

- `1 <= x <= 9`
- 最多调用 `100` 次 `push`、`pop`、`peek` 和 `empty`
- 假设所有操作都是有效的 （例如，一个空的队列不会调用 `pop` 或者 `peek` 操作）

**进阶：**

- 你能否实现每个操作均摊时间复杂度为 `O(1)` 的队列？换句话说，执行 `n` 个操作的总时间复杂度为 `O(n)` ，即使其中一个操作可能花费较长时间。

###### 解答：

```C
typedef struct {
    int stk_in_top,stk_out_top;
    int in[100],out[100];
} MyQueue;

MyQueue* myQueueCreate() {
    MyQueue *queue=(MyQueue*)malloc(sizeof(MyQueue));
    queue->stk_in_top=0;
    queue->stk_out_top=0;
    return queue;
}

void myQueuePush(MyQueue* obj, int x) {
    obj->in[(obj->stk_in_top)++]=x;
}

int myQueuePop(MyQueue* obj) {
    if(obj->stk_out_top==0){
        while(obj->stk_in_top>0){
            obj->stk_in_top--;
            obj->out[obj->stk_out_top]=obj->in[obj->stk_in_top];
            obj->stk_out_top++;
        }
    }
    return obj->out[--obj->stk_out_top];
}

int myQueuePeek(MyQueue* obj) {
    //当stackOut不为空时，stackOut的栈顶元素就是队列的第一个元素。(如果stackOut不为空，说明之前已经将stackIn中的部分或全部元素逆序转移到了stackOut。由于逆序转移的过程，stackOut的栈顶元素始终是最早加入队列的元素。)
    if(obj->stk_out_top==0){
        while(obj->stk_in_top>0){
            obj->stk_in_top--;
            obj->out[obj->stk_out_top]=obj->in[obj->stk_in_top];
            obj->stk_out_top++;
        }
    }
    return obj->out[obj->stk_out_top-1];
}

bool myQueueEmpty(MyQueue* obj) {
    return obj->stk_in_top==0 && obj->stk_out_top==0;
}

void myQueueFree(MyQueue* obj) {
    free(obj);
}

/**
 * Your MyQueue struct will be instantiated and called as such:
 * MyQueue* obj = myQueueCreate();
 * myQueuePush(obj, x);
 
 * int param_2 = myQueuePop(obj);
 
 * int param_3 = myQueuePeek(obj);
 
 * bool param_4 = myQueueEmpty(obj);
 
 * myQueueFree(obj);
*/
```

---

#### 1.26-2 [用队列实现栈](https://leetcode.cn/problems/implement-stack-using-queues/)

请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通栈的全部四种操作（`push`、`top`、`pop` 和 `empty`）。

实现 `MyStack` 类：

- `void push(int x)` 将元素 x 压入栈顶。
- `int pop()` 移除并返回栈顶元素。
- `int top()` 返回栈顶元素。
- `boolean empty()` 如果栈是空的，返回 `true` ；否则，返回 `false` 。

**注意：**

- 你只能使用队列的标准操作 —— 也就是 `push to back`、`peek/pop from front`、`size` 和 `is empty` 这些操作。
- 你所使用的语言也许不支持队列。 你可以使用 list （列表）或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。

**示例：**

```
输入：
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
输出：
[null, null, null, 2, 2, false]

解释：
MyStack myStack = new MyStack();
myStack.push(1);
myStack.push(2);
myStack.top(); // 返回 2
myStack.pop(); // 返回 2
myStack.empty(); // 返回 False
```

**提示：**

- `1 <= x <= 9`
- 最多调用`100` 次 `push`、`pop`、`top` 和 `empty`
- 每次调用 `pop` 和 `top` 都保证栈不为空

**进阶：**你能否仅用一个队列来实现栈。

###### 解答：

```C
//两个队列
typedef struct {
    int *queue1,*queue2;
    int size1,size2;
    int front1,rear1,front2,rear2;
} MyStack;

MyStack* myStackCreate() {
    MyStack *stk=(MyStack*)malloc(sizeof(MyStack));
    stk->queue1=(int *)malloc(sizeof(int)*100);
    stk->queue2=(int *)malloc(sizeof(int)*100);
    stk->front1=stk->rear1=0;
    stk->front2=stk->rear2=0;
    stk->size1=stk->size2=0;
    return stk;
}

void myStackPush(MyStack* obj, int x) {
    obj->queue1[obj->rear1++]=x;
    obj->size1++;
}

int myStackPop(MyStack* obj) {
    while(obj->size1>1){
        obj->queue2[obj->rear2++]=obj->queue1[obj->front1++];
        obj->size2++;
        obj->size1--;
    }
    int ret_element=obj->queue1[obj->front1++];
    obj->size1--;
    //交换队列1和队列2，即将刚刚队列1输出给队列2的元素再全部返回来
    int *temp_queue=obj->queue1;
    obj->queue1=obj->queue2;
    obj->queue2=temp_queue;
    int temp_front=obj->front1;
    obj->front1=obj->front2;
    obj->front2=temp_front;
    int temp_rear=obj->rear1;
    obj->rear1=obj->rear2;
    obj->rear2=temp_rear;
    int temp_size=obj->size1;
    obj->size1=obj->size2;
    obj->size2=temp_size;

    return ret_element;
}

int myStackTop(MyStack* obj) {
    while(obj->size1>1){
        obj->queue2[obj->rear2++]=obj->queue1[obj->front1++];
        obj->size2++;
        obj->size1--;
    }
    int ret_element=obj->queue1[obj->front1];//注意这里obj->rear1不用++

    obj->queue2[obj->rear2++]=obj->queue1[obj->front1++];
    obj->size1--;
    obj->size2++;

    int *temp_queue=obj->queue1;
    obj->queue1=obj->queue2;
    obj->queue2=temp_queue;
    int temp_front=obj->front1;
    obj->front1=obj->front2;
    obj->front2=temp_front;
    int temp_rear=obj->rear1;
    obj->rear1=obj->rear2;
    obj->rear2=temp_rear;
    int temp_size=obj->size1;
    obj->size1=obj->size2;
    obj->size2=temp_size;

    return ret_element;

}

bool myStackEmpty(MyStack* obj) {
    return obj->size1==0 && obj->size2==0;
}

void myStackFree(MyStack* obj) {
    //发生double-free错误的解决方法：
    // if(obj->queue1!=NULL){
    //     free(obj->queue1);
    //     obj->queue1=NULL;
    // }
    // if(obj->queue2!=NULL){
    //     free(obj->queue2);
    //     obj->queue2=NULL;
    // }
    free(obj->queue1);
    free(obj->queue2);
    free(obj);
}

/**
 * Your MyStack struct will be instantiated and called as such:
 * MyStack* obj = myStackCreate();
 * myStackPush(obj, x);
 
 * int param_2 = myStackPop(obj);
 
 * int param_3 = myStackTop(obj);
 
 * bool param_4 = myStackEmpty(obj);
 
 * myStackFree(obj);
*/
```

```C
//一个队列
typedef struct {
    int *queue;
    int size;
    int front;
    int rear;
} MyStack;

MyStack* myStackCreate() {
    MyStack *stk = (MyStack*)malloc(sizeof(MyStack));
    stk->queue = (int *)malloc(sizeof(int) * 100);
    stk->size = 0;
    stk->front = 0;
    stk->rear = 0;
    return stk;
}

void myStackPush(MyStack* obj, int x) {
    obj->queue[obj->rear++]=x;
    obj->size++;
}

int myStackPop(MyStack* obj) {
    if(obj->size==0){
        return -1;
    }
    for(int i=0;i<obj->size-1;i++){
        obj->queue[obj->rear++]=obj->queue[obj->front++];
    }
    int ret_element=obj->queue[obj->front++];
    obj->size--;
    return ret_element;
}

int myStackTop(MyStack* obj) {
    if (obj->size == 0) {
        return -1; 
    }
    for(int i=0;i<obj->size-1;i++){
        obj->queue[obj->rear++]=obj->queue[obj->front++];
    }
    int ret_element=obj->queue[obj->front];
    obj->queue[obj->rear++]=obj->queue[obj->front++];
    return ret_element;
}

bool myStackEmpty(MyStack* obj) {
    return obj->size==0;
}

void myStackFree(MyStack* obj) {
    free(obj->queue);
    free(obj);
}
```

---

#### 1.26-3 [有效的括号](https://leetcode.cn/problems/valid-parentheses/)

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

**示例 1：**

```
输入：s = "()"
输出：true
```

**示例 2：**

```
输入：s = "()[]{}"
输出：true
```

**示例 3：**

```
输入：s = "(]"
输出：false
```

**示例 4：**

```
输入：s = "([])"
输出：true
```

**提示：**

- `1 <= s.length <= 104`
- `s` 仅由括号 `'()[]{}'` 组成

###### 解答：

```C
bool isValid(char* s) {
    char stk[10000];
    int stktop=0;
    //栈顶指针指向栈顶元素的后一个位置，stktop-1位置表示栈顶元素
    for(int i=0;i<strlen(s);i++){
        if(s[i]=='('){
            stk[stktop++]=')';
        }else if(s[i]=='['){
            stk[stktop++]=']';
        }else if(s[i]=='{'){
            stk[stktop++]='}';
        }else{
            //stktop==0表示栈为空但还有括号没有被匹配，s[i]!=stk[--stktop]表示括号不匹配
            if(stktop==0 || s[i]!=stk[--stktop]){
                return false;
            }
        }
    }
    if(stktop!=0){
        //stktop!=0表示栈中还有元素没有被匹配
        return false;
    }else{
        return true;
    }
}
```

---

#### 1.26-4 [删除字符串中的所有相邻重复项](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/)

给出由小写字母组成的字符串 `s`，**重复项删除操作**会选择两个相邻且相同的字母，并删除它们。

在 `s` 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

**示例：**

```
输入："abbaca"
输出："ca"
解释：
例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca"。
```

**提示：**

1. `1 <= s.length <= 105`
2. `s` 仅由小写英文字母组成。

###### 解答：

```C
char* removeDuplicates(char* s) {
    //开辟栈空间，作为最后的返回结果
    char* stk=(char*)malloc(sizeof(char)*(strlen(s)+1));
    int stktop=0;//stktop指向栈顶元素的下一个位置
    int i=0;
    while(i<strlen(s)){
        char temp=s[i];
        if(stktop>0 && temp==stk[stktop-1]){
            //栈内有元素并且temp等于栈顶元素，则进行消除，即stktop--
            stktop--;
        }else{
            stk[stktop++]=temp;
        }
        i++;
    }
    stk[stktop]='\0';
    return stk;
}
```

---

#### 1.26-5 [逆波兰表达式求值]([150. 逆波兰表达式求值 - 力扣（LeetCode）](https://leetcode.cn/problems/evaluate-reverse-polish-notation/description/))

给你一个字符串数组 `tokens` ，表示一个根据 [逆波兰表示法](https://baike.baidu.com/item/逆波兰式/128437) 表示的算术表达式。

请你计算该表达式。返回一个表示表达式值的整数。

**注意：**

- 有效的算符为 `'+'`、`'-'`、`'*'` 和 `'/'` 。
- 每个操作数（运算对象）都可以是一个整数或者另一个表达式。
- 两个整数之间的除法总是 **向零截断** 。
- 表达式中不含除零运算。
- 输入是一个根据逆波兰表示法表示的算术表达式。
- 答案及所有中间计算结果可以用 **32 位** 整数表示。

**示例 1：**

```
输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
```

**示例 2：**

```
输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
```

**示例 3：**

```
输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
解释：该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

**提示：**

- `1 <= tokens.length <= 104`
- `tokens[i]` 是一个算符（`"+"`、`"-"`、`"*"` 或 `"/"`），或是在范围 `[-200, 200]` 内的一个整数

**逆波兰表达式：**

逆波兰表达式是一种后缀表达式，所谓后缀就是指算符写在后面。

- 平常使用的算式则是一种中缀表达式，如 `( 1 + 2 ) * ( 3 + 4 )` 。
- 该算式的逆波兰表达式写法为 `( ( 1 2 + ) ( 3 4 + ) * )` 。

逆波兰表达式主要有以下两个优点：

- 去掉括号后表达式无歧义，上式即便写成 `1 2 + 3 4 + * `也可以依据次序计算出正确结果。
- 适合用栈操作运算：遇到数字则入栈；遇到算符则取出栈顶两个数字进行计算，并将结果压入栈中

###### 解答：

```C
//判断字符s是否为运算符
int isOperator(char *s){
    return !strcmp(s,"+") || !strcmp(s,"-") || !strcmp(s,"*") || !strcmp(s,"/");
}
int evalRPN(char** tokens, int tokensSize) {
    //用于存放操作数的整型类型的栈
    int* stk=(int*)malloc(sizeof(int)*tokensSize);
    int stktop=0;
    for(int i=0;i<tokensSize;i++){
        if(!isOperator(tokens[i])){
            //若不是运算符，即若为操作数，则将字符类型转换为int型再压入栈中，atoi函数可以将字符串str转换为一个整数
            stk[stktop++]=atoi(tokens[i]);
        }else{
            //若为运算符，则弹出栈中两个操作数并进行运算，再将结果压入栈中
            int num1=stk[--stktop];
            int num2=stk[--stktop];
            //switch 语句的表达式必须是 整数类型（int、char 等），而不能是字符串（char*），题目中给出的指针是char** tokens，所以tokens[i][0]才表示当前遍历到的字符元素，而tokens[i]表示当前字符的指针
            switch(tokens[i][0]){
                case '+':
                    //注意两个操作数的运算顺序是num2在前，num1在后
                    stk[stktop++]=num2+num1;
                    break;
                case '-':
                    stk[stktop++]=num2-num1;
                    break;
                case '*':
                    stk[stktop++]=num2*num1;
                    break;
                case '/':
                    stk[stktop++]=num2/num1;
                    break;
            }
        }
    }
    int result=stk[stktop-1];
    free(stk);
    return result;
}
```

---

#### 1.26-6 [滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回 *滑动窗口中的最大值* 。

**示例 1：**

```
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**示例 2：**

```
输入：nums = [1], k = 1
输出：[1]
```

**提示：**

- `1 <= nums.length <= 105`
- `-104 <= nums[i] <= 104`
- `1 <= k <= nums.length`

###### 解答：

```C
/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* maxSlidingWindow(int* nums, int numsSize, int k, int* returnSize) {
    int* result=(int*)malloc(sizeof(int)*(numsSize-k+1));
    *returnSize=numsSize-k+1;
    int* deque=(int*)malloc(sizeof(int)*numsSize);
    int dequeSize=0;
    for (int i = 0; i < numsSize; i++) {
        // 移除队列中不在当前窗口范围内的索引
        if (dequeSize > 0 && deque[0] < i - k + 1) {
            dequeSize--;
        }

        // 移除队列中所有小于当前元素的索引
        while (dequeSize > 0 && nums[deque[dequeSize - 1]] < nums[i]) {
            dequeSize--;
        }

        // 将当前元素的索引加入队列
        deque[dequeSize++] = i;

        // 记录当前窗口的最大值
        if (i >= k - 1) {
            result[i - k + 1] = nums[deque[0]];
        }
    }

    free(deque); // 释放单调队列
    return result;
}
```

---

#### 1.26-7 [前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

**示例 1:**

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

**示例 2:**

```
输入: nums = [1], k = 1
输出: [1]
```

**提示：**

- `1 <= nums.length <= 105`
- `k` 的取值范围是 `[1, 数组中不相同的元素的个数]`
- 题目数据保证答案唯一，换句话说，数组中前 `k` 个高频元素的集合是唯一的

**进阶：**你所设计算法的时间复杂度 **必须** 优于 `O(n log n)` ，其中 `n` 是数组大小。

###### 解答：

```C
                                                        /*大小根堆切换开关*/
// int  HeapLen(Heap* hp);                                 //heap获取当前的堆大小
// void HeapSwap(int* pLeft, int* pRight);                 //heap交换父子结点的数值
// bool HeapEmpty(Heap* hp);                               //heap判空
// bool HeapFull(Heap* hp);                                //heap判满
// int  HeapGetTop(Heap* hp);                              //heap获取堆顶
// void HeapInsert(Heap* hp, int  dat,int value,bool isMax);//heap向堆的尾部插入1个元素
// void HeapDelete(Heap* hp,bool isMax);                    //heap删除堆顶
// void HeapAdjustDown(Heap* hp, int parent,bool isMax);   //heap向下调整
// void HeapAdjustUp(Heap* hp, int child,bool isMax);      //heap向上调整
// Heap* CreateHeap(int size);                             //heap创建
// void heapFree(Heap* hp);                                //heap释放空间
typedef struct Heap
{
    int* array;   //存放堆的数组
    int* value;
    int  capacity;//数组的容量
    int  len;     //已存数组的大小
}Heap;

int HeapLen(Heap* hp)
{
    return hp->len;
}

bool HeapEmpty(Heap* hp)          //判空
{
    if (HeapLen(hp) == 1)
    {
        return true;
    }
    return false;
}

bool HeapFull(Heap* hp)          //判满
{
    if (hp->capacity == hp->len)
    {
        return true;
    }
    return false;
}

void HeapSwap(int* pLeft, int* pRight)//交换数值
{
    //交换堆中的父子结点
    int  temp;
    temp    = *pLeft;
    *pLeft  = *pRight;
    *pRight = temp;
}

int HeapGetTop(Heap* hp)
{
    return hp->array[1];
}

Heap* CreateHeap(int size)
{
    Heap* heap = (Heap*)malloc(sizeof(Heap));
    int   heapLen = size + 1;//长度比size的长度大1才行
    //给堆申请空间,初始化
    heap->array = (int*)malloc(sizeof(int) * heapLen);
    heap->value = (int*)malloc(sizeof(int) * heapLen);
    heap->capacity  = heapLen;     //容量
    heap->len       = 1;           //当前大小
    return heap;
}

void HeapAdjustDown(Heap* hp, int parent ,bool isMax)//向下调整
{
    //标记左右孩子中最小孩子
    int child = 2 * parent;            //左孩子为2*parent  右孩子为 2*parent +1
    int len  = hp->len;
    while (child < len)
    {
        if(isMax)
        {
            //大根堆 选最大的
            //有右子树时 ，找左右孩子中最大的孩子 
            if ((child + 1 < len) && hp->array[child] < hp->array[child + 1])
                child = child + 1;

            //最大孩子大于双亲时 ，孩子与双亲数值交换，否则说明已经调好，不用继续
            if (hp->array[child] > hp->array[parent])
            {
                HeapSwap(&hp->array[child], &hp->array[parent]);
                HeapSwap(&hp->value[child], &hp->value[parent]);
                parent = child;
                child = parent << 1;
            }
            else
                return;
        }
        else
        {
            //小根堆  选最小的
            //有右子树时 ，找左右孩子中最小的孩子 
            if ((child + 1 < len) && hp->array[child] > hp->array[child + 1])
                child = child + 1;

            //最小孩子小于双亲时 ，孩子与双亲数值交换，否则说明已经调好，不用继续
            if (hp->array[child] < hp->array[parent])
            {
                HeapSwap(&hp->array[child], &hp->array[parent]);
                HeapSwap(&hp->value[child], &hp->value[parent]);
                parent = child;
                child = parent << 1;
            }
            else
                return;
        }
    }
}

void HeapAdjustUp(Heap* hp, int child,bool isMax)//向上调整
{
    //得到父母结点的位置
    int parent = child / 2;
    while (child > 1)
    {
        if(isMax)
        {
            //大根堆选择大的
            //循环迭代从child当前位置一直迭代到0位置即对顶
            if (hp->array[child] > hp->array[parent])
            {
                HeapSwap(&hp->array[child], &hp->array[parent]);
                HeapSwap(&hp->value[child], &hp->value[parent]);
                child = parent;
                parent = child/2;
            }
            else
                return;
        }
        else
        {
            //小根堆选择小的
            //循环迭代从child当前位置一直迭代到0位置即对顶
            if (hp->array[child] < hp->array[parent])
            {
                HeapSwap(&hp->array[child], &hp->array[parent]);
                HeapSwap(&hp->value[child], &hp->value[parent]);
                child = parent;
                parent = child/2;
            }
            else
                return;
        }
    }
}

void HeapDelete(Heap* hp,bool isMax)//删除堆顶
{
    if (HeapEmpty(hp))
        return;

    //用最后一个元素覆盖堆顶，相当于删除堆顶
    hp->array[1] = hp->array[hp->len - 1];
    hp->value[1] = hp->value[hp->len - 1];
    hp->len--;//删除最后一个元素 heap长度变短
    HeapAdjustDown(hp, 1,isMax);//对第一个元素进行调整
}

void HeapInsert(Heap* hp, int  dat,int value,bool isMax)
{
    if (HeapFull(hp))
    {
        //扩容
        hp->capacity <<= 1; 
        hp->array = (int *) realloc(hp->array, hp->capacity * sizeof(int));
    }
    int child = 0;
    int parent = 0;
    //插入到最后一个元素的下一个位置
    hp->array[hp->len] = dat;
    hp->value[hp->len] = value;
    hp->len++;
    //调整刚插入元素，
    //因为插入的是堆的尾部，需要堆向上调整
    HeapAdjustUp(hp, hp->len - 1,isMax);
}


void heapFree(Heap* hp)
{
    free(hp->array);
    free(hp->value);
    free(hp);
}

#define MAXSIZE 769/* 选取一个质数即可 */
typedef struct Node 
{
    int key;
    int value;
    struct Node *next;  //保存链表表头
} List;

typedef struct 
{
    List *hashHead[MAXSIZE];//定义哈希数组的大小
} MyHashMap;

List * isInHash(List *list, int key) 
{
    List *nodeIt = list;
    //通过链表下遍历
    while (nodeIt != NULL) 
    {
        if (nodeIt->key == key) 
        {
            return nodeIt;
        }
        nodeIt = nodeIt->next;
    }
    return NULL;
}

MyHashMap* myHashMapCreate() 
{
    int i;
    MyHashMap* newHash= (MyHashMap* )malloc(sizeof(MyHashMap));
    /* 对链表的头结点赋初值 */
    for (i = 0; i < MAXSIZE; i++)
    {
        newHash->hashHead[i] = NULL;
    }
    return newHash;
}

void myHashMapPut(MyHashMap* obj, int key, int value) 
{
    List * it= isInHash(obj->hashHead[abs(key)%MAXSIZE],key);
    if(it != NULL)
    {
        //在表里面更新键值
        it->value = value;
    }
    else
    {
        //不在表里面
        List *newNode   = (List*)malloc(sizeof(List));
        newNode->key    = key;
        newNode->next   = NULL;
        newNode->value  = value;
        if(obj->hashHead[abs(key)%MAXSIZE] != NULL)
        {
            //当前头链表不为空，则需要将后续的链表接上
            //需要主要这里表头也代表一个数据的值
            newNode->next = obj->hashHead[abs(key)%MAXSIZE];
        }
        //修改头链表
        obj->hashHead[abs(key)%MAXSIZE] =  newNode;
    }
}

int myHashMapGet(MyHashMap* obj, int key) 
{
    List * it= isInHash(obj->hashHead[abs(key)%MAXSIZE],key);
    if( it!= NULL)
    {
        return it->value;
    }
    return -1;
}

void myHashMapRemove(MyHashMap* obj, int key) 
{    
    List *preIt = NULL;
    List *curIt = obj->hashHead[abs(key)%MAXSIZE];
    //通过链表下遍历
    while (curIt != NULL) 
    {
        if (curIt->key == key) 
        {
            break;
        }
        preIt = curIt;
        curIt = curIt->next;
    }

    if(curIt == NULL)
    {
        //没有找到
        return;
    }

    //找到了
    if(preIt == NULL)
    {
        //等于表头
        obj->hashHead[abs(key)%MAXSIZE] = curIt->next;
    }
    else
    {

        preIt->next = curIt->next;
    }
    free(curIt);
}

void myHashMapFree(MyHashMap* obj) 
{
   int i;
   List *freeIt;
   List *curIt;
   for (i = 0; i < MAXSIZE; i++)
    {
        if(obj->hashHead[i] != NULL)
        {
            freeIt = NULL;
            curIt  = obj->hashHead[i];
            
            while(curIt != NULL)
            {
                freeIt = curIt;
                curIt= curIt->next;
                free(freeIt);
            }
            obj->hashHead[i]= NULL;
        }
    }
    free(obj);
}

int* topKFrequent(int* nums, int numsSize, int k, int* returnSize)
{
    //hsahSet + 最小堆
    
    #if 0 //有负数直接挂掉了
    int  hashSet[100001] = {0};
    //hash处理
    for(int i = 0; i< numsSize; i++)
    {
        hashSet[nums[i]]++;
    }
    #endif



    MyHashMap * hashMap = myHashMapCreate();
    for(int i = 0; i< numsSize; i++)
    {
        int count = myHashMapGet(hashMap, nums[i]);
        if(count == -1)
        {
            myHashMapPut(hashMap,nums[i], 1);
        }
        else
        {
            myHashMapPut(hashMap,nums[i], count + 1);
        }
    }
    //申请k个大小的堆
    Heap* newHeap = CreateHeap(k);

    List *freeIt;
    List *curIt;
    for (int  i = 0; i < MAXSIZE; i++)
    {
        if(hashMap->hashHead[i] != NULL)
        {
            freeIt = NULL;
            curIt  = hashMap->hashHead[i];
            
            while(curIt != NULL)
            {
                freeIt = curIt;
                curIt= curIt->next;

                //堆处理
                if(HeapFull(newHeap) == false)
                {
                    //没满直接添加
                    HeapInsert(newHeap,freeIt->value,freeIt->key,false);
                }
                else
                {
                    //满了和堆顶比较
                    if(newHeap->array[1] < freeIt->value)
                    {
                        //修改堆顶值
                        newHeap->array[1] = freeIt->value;
                        newHeap->value[1] = freeIt->key;
                        HeapAdjustDown(newHeap, 1, false);
                    } 
                }
                free(freeIt);
            }
            hashMap->hashHead[i]= NULL;
        }
    }
    free(hashMap);
    //直接遍历堆 因为堆使用数组存放的
    *returnSize = k;
    int* returnNums = (int *)malloc(sizeof(int) * k);
    for(int i = 0; i< k; i++)
    {
        returnNums[i] = newHeap->value[i+1];
    }
    heapFree(newHeap);
    return returnNums;
}
```

