from collections import defaultdict
from datetime import datetime
import pandas as pd
import json

treatment_name_mapping = {
    'conversation_5': 'conversation_starter',
    'chat_5': 'chat',
    'feedback_5': 'feedback',
    'suggestions_5': 'suggestions',
    'baseline_5': 'baseline',
}


class ConversationTree:
    def __init__(self, data):
        # Metadata for all rounds
        self.round_starts = data.get("round_starts", [])
        self.topic_order = data.get("topic_order", [])
        self.treatment_name = data.get("treatmentName")

        
        # Create separate tree structures for each round
        self.rounds = {}  # Dictionary of round_idx -> RoundTree
        
        # Load seed content first
        with open('src/genai-social-media/dashboard/treatments/seed_content.json', 'r') as f:
            seed_content = json.load(f)
        
        # Build separate trees for each round
        for round_idx, round_comments in enumerate(data["comments"]):
            topic = self.topic_order[round_idx]
            # Initialize RoundTree with seed content for this topic
            self.rounds[round_idx] = RoundTree(
                seed_content.get(topic, []),  # Get seed content for this topic
                start_time=self.round_starts[round_idx],
                topic=topic,
                treatment_name=self.treatment_name,
                treatment_content=data.get(treatment_name_mapping[self.treatment_name], {}).get(round_idx, {})
            )
            # Add game comments
            for comment in round_comments:
                self.rounds[round_idx].add_comment(comment)
            
            # Add actions for this round
            round_actions = data.get("actions", [])[round_idx]
            for action in round_actions:
                self.rounds[round_idx].add_action(action)

class RoundTree:
    def __init__(self, seed_data, start_time, topic, treatment_name, treatment_content):
        self.tree = defaultdict(list)
        self.treatment_name = treatment_name
        self.parent_map = {}
        self.user_map = {}
        self.timestamp_map = {}
        self.content_map = {}
        self.start_time = pd.Timestamp(start_time)
        self.topic = topic
        self.treatment_content = treatment_content
        self.starter_map = {}
        self.seed_content_ids = set()
        self.actions_map = defaultdict(list)  # Maps content_id to list of actions
        
        # Load seed content first
        for item in seed_data:
            self._add_comment(item, is_seed=True)
    
    def _add_comment(self, item, is_seed=False):
        content_id = item["content_id"]
        parent_id = item["parent_content_id"]
        user_id = item["user_id"]
        timestamp = pd.Timestamp(item["timestamp"])
        content = item["content"]
        
        # Basic tree structure
        self.tree[parent_id].append(content_id)
        self.parent_map[content_id] = parent_id
        self.user_map[content_id] = user_id
        self.timestamp_map[content_id] = timestamp
        self.content_map[content_id] = content
        
        if is_seed:
            self.seed_content_ids.add(content_id)
    
    def count_descendants(self, content_id):
        """Returns the number of nodes below a given node."""
        def count_nodes(node):
            return 1 + sum(count_nodes(child) for child in self.tree[node])
        return count_nodes(content_id) - 1
    
    def has_descendants(self, content_id):
        """Returns True if the node has any descendants, False otherwise."""
        return bool(self.tree[content_id])
    
    def get_depth(self, content_id):
        """
        Returns the depth of a node in the tree, stopping at seed content.
        Depth is counted only through non-seed content.
        """
        depth = 0
        while content_id in self.parent_map:
            content_id = self.parent_map[content_id]
            if content_id in self.seed_content_ids:
                break
            depth += 1
        return depth
    
    def count_unique_users_above(self, content_id):
        """
        Returns the number of unique user_ids who posted before this content's timestamp,
        excluding seed content.
        """
        if content_id not in self.timestamp_map:
            return 0
        
        content_timestamp = self.timestamp_map[content_id]
        unique_users = set()
        
        # Check all nodes in the tree
        for node_id in self.parent_map.keys():
            # Skip seed content and the current content
            if node_id in self.seed_content_ids or node_id == content_id:
                continue
            
            # If the node's timestamp is before our content's timestamp
            if (node_id in self.timestamp_map and 
                self.timestamp_map[node_id] < content_timestamp):
                unique_users.add(self.user_map[node_id])
            
        return len(unique_users)
    
    def count_nodes_before(self, content_id):
        """
        Returns the total number of nodes that were posted before this content's timestamp,
        excluding seed content.
        """
        if content_id not in self.timestamp_map:
            return 0
            
        content_timestamp = self.timestamp_map[content_id]
        count = 0
        
        # Check all nodes in the tree
        for node_id in self.parent_map.keys():
            # Skip seed content and the current content
            if node_id in self.seed_content_ids or node_id == content_id:
                continue
                
            # If the node's timestamp is before our content's timestamp
            if (node_id in self.timestamp_map and 
                self.timestamp_map[node_id] < content_timestamp):
                count += 1
                
        return count
    
    def time_left(self, content_id):
        """
        Computes the time remaining until 10 minutes after round start time.
        Returns the number of seconds remaining (float).
        Negative values indicate the comment was made after the round ended.
        """
        if content_id not in self.timestamp_map:
            return None
        
        comment_timestamp = self.timestamp_map[content_id]
        round_end_time = self.start_time + pd.Timedelta(minutes=10)
        
        # Convert timedelta to seconds for numerical analysis
        return (round_end_time - comment_timestamp).total_seconds() / 600  # Scale to be between 0 and 1
    
    def get_nodes(self):
        """Retrieves the list of nodes in the tree, excluding seed content."""
        return [node for node in self.parent_map.keys() 
                if node not in self.seed_content_ids]
    
    def get_content(self, content_id):
        """Retrieves the content of a specific node."""
        return self.content_map.get(content_id)
    
    def get_starter_used(self, content_id):
        """Returns the conversation starter used for this content, if any."""
        return self.starter_map.get(content_id)

    # Public method for adding game comments
    def add_comment(self, item):
        self._add_comment(item, is_seed=False)

    def was_treatment_used(self, content_id):
        """
        Check if a specific comment was made after the user used a conversation starter
        Returns:
        - True/False: whether the user used a starter for that comment
        - List of starters used (empty if none used) or for suggestions_5, a list of booleans indicating which type was used [agreeing, neutral, disagreeing]
        """
        if content_id not in self.parent_map:
            return False, []
        
        # Get the comment's metadata
        user_id = self.user_map.get(content_id)
        parent_id = self.parent_map.get(content_id)
        comment_timestamp = self.timestamp_map.get(content_id)
        parent_content = self.content_map.get(parent_id)
        comment_content = self.content_map.get(content_id)
        
        if not all([user_id, parent_id, comment_timestamp, parent_content]):
            return False, [], comment_content, parent_content
        
        # Get all starters used by this user
        treatment_content_user = self.treatment_content.get(user_id, [])
        
        # Find matching starters where:
        # 1. The starter was used on this parent content
        # 2. The starter was used before the comment was made
        if self.treatment_name == "conversation_5":
            matching_starters = [
                starter for starter in treatment_content_user
                if (starter[0] == parent_content and  # Content matches parent
                    pd.Timestamp(starter[2]) < comment_timestamp)  # Starter used before comment
            ]
            return bool(matching_starters), matching_starters, comment_content, parent_content

        elif self.treatment_name == "chat_5":
            # Check if any interaction happened before the comment
            for interaction in treatment_content_user:
                interaction_timestamp = pd.Timestamp(interaction[1])
                # Check if interaction happened within 2 minutes before the comment
                time_diff = (comment_timestamp - interaction_timestamp).total_seconds()
                if 0 <= time_diff <= 120:  # 120 seconds = 2 minutes
                    return True, []
            
            return False, []
            
        elif self.treatment_name == "feedback_5":            
            # Check if there's feedback for the parent of this comment
            if str(parent_id) in treatment_content_user:
                # Get all feedback instances for this parent
                feedback_instances = treatment_content_user[str(parent_id)]
                
                for feedback_instance in feedback_instances:
                    # The timestamp is the last element in the tuple
                    feedback_timestamp = pd.Timestamp(feedback_instance[2])
                    
                    # Check if feedback was given before the comment
                    if feedback_timestamp < comment_timestamp:
                        return True, []
            
            return False, []
                        
        elif self.treatment_name == "suggestions_5":
            # Default result: no suggestion type was used
            result = [False, False, False]  # [agreeing, neutral, disagreeing]
            
            # Check if user has suggestions data
            if not treatment_content_user or 'selected' not in treatment_content_user:
                return False, result
                
            # Get the selected suggestions
            selected_suggestions = treatment_content_user.get('selected', [])
            
            # Get the suggestions list
            suggestions_list = treatment_content_user.get('suggestions', [])
            
            # Check each suggestion to see if it matches the parent content
            for suggestion in suggestions_list:
                suggestion_parent_content = suggestion[0]  # First element is parent content
                suggestion_options = suggestion[1]  # Second element is list of reply options
                suggestion_timestamp = pd.Timestamp(suggestion[2])  # Third element is timestamp
                
                # If this suggestion is for the parent of our comment and was shown before the comment
                if suggestion_parent_content == parent_content and suggestion_timestamp < comment_timestamp:
                    # Check if any selected suggestion was used for this comment
                    for selected in selected_suggestions:
                        selected_text = selected[0]  # The text of the selected suggestion
                        selected_timestamp = pd.Timestamp(selected[1])
                        
                        # If the selection was made before the comment
                        if selected_timestamp < comment_timestamp:
                            # Check which type of suggestion was selected
                            for i, option in enumerate(suggestion_options):
                                # The option is a dictionary with a 'reply' key
                                if 'reply' in option and option['reply'] == selected_text:
                                    # Mark the corresponding type as used
                                    result[i] = True
                                    return True, result
            
            return False, result

    def add_action(self, action):
        """Add an action/reaction to the specified content"""
        content_id = action["content_id"]
        self.actions_map[content_id].append({
            'user_id': action["user_id"],
            'like_type': action["like_type"],
            'timestamp': pd.Timestamp(action["timestamp"])
        })

    def count_actions(self, content_id):
        """Returns the total number of actions/reactions for a given content"""
        return len(self.actions_map.get(content_id, []))

    def has_reactions(self, content_id):
        """Returns True if the content has received any reactions"""
        return content_id in self.actions_map and len(self.actions_map[content_id]) > 0

    def count_action_types(self, content_id):
        """
        Returns a dictionary with counts of each reaction type for a given content
        
        Args:
            content_id: The ID of the content to count reactions for
            
        Returns:
            list: List of counts for each action type (e.g., [3, 2, 0, 1, 0, 0, 0])
        """
        
        # Get all actions for this content
        actions = self.actions_map.get(content_id, [])

        actions_types = ['Like', 'Love', 'Dislike', 'Wow', 'Sad', 'Angry', 'Funny']
        action_counts = [0] * len(actions_types)

        
        # Count each action type
        for action in actions:
            like_type = action.get('like_type')
            if like_type:
                action_counts[actions_types.index(like_type)] += 1
                
        return action_counts
    
    def count_direct_children(self, content_id):
        """
        Returns the number of comments that have this content_id as their parent
        
        Args:
            content_id: The ID of the content to count children for
            
        Returns:
            int: Number of direct child comments
        """
        # self.tree is a defaultdict(list) that maps parent_id to list of child content_ids
        return len(self.tree.get(content_id, []))

    def get_content_length(self, content_id):
        """
        Returns the length (character count) of the content for a given content_id
        
        Args:
            content_id: The ID of the content to measure
            
        Returns:
            int: Number of characters in the content, or 0 if content_id not found
        """
        content = self.content_map.get(content_id)
        if content is None:
            return 0
        
        return len(content.split())
