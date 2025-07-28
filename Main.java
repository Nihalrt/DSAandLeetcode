import java.util.*;

public class Main {

    public static void main(String[] args){

        String test = "abcabcbb";
        int GetMax = lengthOfLongestSubstring(test);
        System.out.println(GetMax);

    }

    /*
     * Solved Longest Substring Without Repeating Characters problem using a HashMap
     * The idea is to use a sliding window approach with two pointers (left and right).
     * We maintain a HashMap to keep track of the characters and their latest indices.
     * As we iterate through the string with the right pointer, we check if the character is already in the HashMap.
     * If it is, we move the left pointer to the right of the last occurrence of that character to ensure that we do not have repeating characters in the current substring.
     * We then update the HashMap with the current character's index and calculate the length of the current substring.
     * We keep track of the maximum length found so far.
     * Time Complexity: O(n) where n is the length of the string, as we traverse the string once.
     * Space Complexity: O(min(n, m)) where n is the length of the string and m is the size of the character set (e.g., 26 for lowercase English letters).
     * This is because we store at most m characters in the HashMap at any time.
     */
    public static int lengthOfLongestSubstring(String s) 
    {
        HashMap<Character, Integer> tracker = new HashMap<>();
        int maxlength = 0;
        int left = 0;

        for (int right = 0; right < s.length(); right++)
        {
            char current = s.charAt(right);

            if(tracker.containsKey(current))
            {
                left = Math.max(tracker.get(current)+1, left);
            }
            tracker.put(current, right);
            maxlength = Math.max(maxlength, right - left + 1);


        }
        return maxlength;


        
    }


    // Solved Partition problem using a Greedy Approach
    // The idea is to sort the array and then iterate through it, keeping track of the
    // minimum value of the current subarray. If the difference between the current number
    // and the minimum value of the current subarray is greater than k, we start a new subarray.
    // This way, we can count the number of subarrays that can be formed such that the difference
    // between the maximum and minimum values in each subarray is at most k.
    // Time Complexity: O(n log n) due to sorting, where n is the number of elements in the array.
    // Space Complexity: O(1) if we ignore the input array, or O(n) if we consider the space used for sorting.
    public static int partitionArray(int[] nums, int k) {
        Arrays.sort(nums);

        int sub_count=0;
        int min_val = -1;

        for(int num: nums){

            if(sub_count == 0 || (num - min_val) > k){
                sub_count++;
                min_val = num;
            }


        }
        return sub_count;
        
    }

    // Solved Content Children (Cookies) using the greedy approach (greedy choice: least greedy child)
    // We sort both the children's greed factor and the cookie sizes.
    // We then iterate through both arrays, giving the smallest cookie that can satisfy the child's greed
    // If a cookie can satisfy a child's greed, we increment the count of satisfied children and move to the next child.
    // If a cookie cannot satisfy the child's greed, we move to the next cookie.
    // The process continues until we either run out of cookies or children.
    // Time Complexity: O(n log n) due to sorting, where n is the number of children or cookies.
    // Space Complexity: O(1) if we ignore the input arrays
    // or O(n) if we consider the space used for sorting.
    public static int ContentChildren(int[] g, int[] s){
        Arrays.sort(g);
        Arrays.sort(s);
        int st_idx = 0;
        int c_idx = 0;
        int give_c = 0;

        while (st_idx < g.length && c_idx < s.length){
            if (s[c_idx] >= g[st_idx]){
                give_c++;
                st_idx++;
            }
            c_idx++;
        }
        return give_c;
    }
    /* Solved Longest Consecutive Sequence problem using a HashSet
     * Basically, we use the Hashset to store all the numbers in the array (Remove duplicates)
     * We check whether the current number is the start of a sequence by checking if the previous number (current_num -1) 
       is already in the set. If yes, we skip to the next number. If not, we start counting the streak from the current number and keep checking (while condition)for the next consecutive number (current_num + 1) in the set.
       Once we reach the end of the streak, we compare the current streak with the longest streak found so far and update it if necessary.
       Finally, we return the longest streak found. 
       Time Complexity: O(n) where n is the number of elements in the array
       Space Complexity: O(n) for the HashSet to store the elements. 
     */
    public int longestConsecutive(int[] nums) {
        HashSet<Integer> checker = new HashSet<>();
        int longest_streak = 0;
        int current_num = 0;
        int current_streak = 0;
        for (int i=0; i < nums.length; i++)
        {
            checker.add(nums[i]);
        }

        for (int num: checker){     
            if(!checker.contains(num-1)){
                current_num = num;
                current_streak = 1;
                while(checker.contains(current_num+1)){
                    current_num+=1;
                    current_streak+=1;
                }
            }   
            longest_streak = Math.max(longest_streak, current_streak);  


        }
        return longest_streak;
    
}

 

    





    
}
